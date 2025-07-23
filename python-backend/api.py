from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import re
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import rag_store

load_dotenv()  # Carrega variáveis de ambiente do arquivo .env

from main import (
    triage_agent,
    faq_agent,
    seat_booking_agent,
    flight_status_agent,
    cancellation_agent,
    portfolio_agent,
    create_initial_context,
)

from agents import (
    Runner,
    ItemHelpers,
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    InputGuardrailTripwireTriggered,
    Handoff,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration (adjust as needed for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class MessageResponse(BaseModel):
    content: str
    agent: str

class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []

# =========================
# In-memory store for conversation state
# =========================

class ConversationStore:
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        pass

    def save(self, conversation_id: str, state: Dict[str, Any]):
        pass

class InMemoryConversationStore(ConversationStore):
    _conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state

# TODO: when deploying this app in scale, switch to your own production-ready implementation
conversation_store = InMemoryConversationStore()

# =========================
# Helpers
# =========================

def extract_mention(text: str) -> str | None:
    match = re.search(r'@([a-zA-Z0-9_]+)', text)
    if match:
        return match.group(1)
    return None

def _get_agent_by_name(name: str):
    """Return the agent object by name."""
    agents = {
        triage_agent.name: triage_agent,
        faq_agent.name: faq_agent,
        seat_booking_agent.name: seat_booking_agent,
        flight_status_agent.name: flight_status_agent,
        cancellation_agent.name: cancellation_agent,
        portfolio_agent.name: portfolio_agent,
    }
    return agents.get(name, triage_agent)

def _get_guardrail_name(g) -> str:
    """Extract a friendly guardrail name."""
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    return str(g)

def _build_agents_list() -> List[Dict[str, Any]]:
    """Build a list of all available agents and their metadata."""
    def make_agent_dict(agent):
        return {
            "name": agent.name,
            "description": getattr(agent, "handoff_description", ""),
            "handoffs": [getattr(h, "agent_name", getattr(h, "name", "")) for h in getattr(agent, "handoffs", [])],
            "tools": [getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(agent, "tools", [])],
            "input_guardrails": [_get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])],
        }
    return [
        make_agent_dict(triage_agent),
        make_agent_dict(portfolio_agent),
        make_agent_dict(faq_agent),
        make_agent_dict(seat_booking_agent),
        make_agent_dict(flight_status_agent),
        make_agent_dict(cancellation_agent),
    ]

# =========================
# Endpoints
# =========================

from fastapi import Request
from typing import Literal

# OpenAI-compatible models endpoint
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "sig9-agent-0.1",
                "object": "model",
                "created": 0,
                "owned_by": "sig9",
            }
        ]
    }

# OpenAI-compatible chat completions endpoint
class OpenAIMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: Optional[bool] = False

class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str = "stop"

class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]

@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_chat_completions(req: OpenAIChatRequest, request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Agora delega internamente para o mesmo fluxo do /chat, garantindo lógica idêntica à UI React.
    """
    import uuid
    import datetime
    import json

    # Loga o payload recebido
    try:
        payload = request._json if hasattr(request, "_json") else req.dict()
        print("[DEBUG] Payload recebido em /v1/chat/completions:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[DEBUG] Falha ao logar payload: {e}")

    # Validação básica do payload
    if not hasattr(req, "messages") or not isinstance(req.messages, list):
        print("[ERROR] Payload inválido: campo 'messages' ausente ou mal formatado.")
        return OpenAIChatResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(datetime.datetime.utcnow().timestamp()),
            model=getattr(req, "model", "sig9-agent-0.1"),
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content="Erro: payload inválido, campo 'messages' ausente."),
                    finish_reason="stop"
                )
            ]
        )

    # Tenta obter conversation_id do header ou body (OpenAI API não define, mas pode ser custom)
    conversation_id = request.headers.get("Conversation-Id") or getattr(req, "conversation_id", None)

    # Filtra mensagens do tipo "assistant" que sejam fallback de erro
    FALLBACK_RESPONSES = [
        "Desculpe, não consegui gerar uma resposta.",
        "Sorry, I can only answer questions related to airline travel."
    ]
    # Detecta se é um prompt de meta-função do OpenWebUI (ex: começa com "### Task:")
    try:
        if req.messages and req.messages[-1].role == "user" and isinstance(req.messages[-1].content, str) and req.messages[-1].content.strip().startswith("### Task:"):
            prompt = req.messages[-1].content.strip()
            print("[DEBUG] Prompt de meta-função detectado, respondendo customizado para compatibilidade OpenWebUI.")
            # Follow-ups
            if "Suggest 3-5 relevant follow-up questions" in prompt:
                follow_ups = [
                    "Você pode me explicar melhor?",
                    "Quais são os próximos passos?",
                    "Pode dar um exemplo?",
                    "Como isso se aplica ao meu caso?",
                    "Existe documentação sobre isso?"
                ]
                content = '{ "follow_ups": ' + str(follow_ups).replace("'", '"') + ' }'
            # Title
            elif "Generate a concise, 3-5 word title" in prompt:
                content = '{ "title": "💬 Conversa com o agente" }'
            # Tags
            elif "Generate 1-3 broad tags" in prompt:
                content = '{ "tags": ["General", "Atendimento", "Chatbot"] }'
            else:
                content = ""
            return OpenAIChatResponse(
                id=f"cmpl-{uuid.uuid4().hex}",
                created=int(datetime.datetime.utcnow().timestamp()),
                model=getattr(req, "model", "sig9-agent-0.1"),
                choices=[
                    OpenAIChatChoice(
                        index=0,
                        message=OpenAIMessage(role="assistant", content=content),
                        finish_reason="stop"
                    )
                ]
            )
    except Exception as e:
        print(f"[ERROR] Falha ao processar prompt de meta-função: {e}")
        return OpenAIChatResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(datetime.datetime.utcnow().timestamp()),
            model=getattr(req, "model", "sig9-agent-0.1"),
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content="Erro ao processar comando especial do OpenWebUI."),
                    finish_reason="stop"
                )
            ]
        )
    # Robustez: filtra apenas mensagens válidas e ignora campos malformados
    # Para simular exatamente o fluxo do /chat, use ChatRequest e chame chat_endpoint
    try:
        # Extrai a última mensagem do usuário (ignora comandos especiais)
        user_msgs = [m for m in req.messages if hasattr(m, "role") and m.role == "user" and not (isinstance(m.content, str) and m.content.strip().startswith("### Task:"))]
        last_user_msg = user_msgs[-1].content if user_msgs else ""
        # Usa o mesmo fluxo do /chat
        chat_req = ChatRequest(conversation_id=None, message=last_user_msg)
        chat_resp = await chat_endpoint(chat_req)
        # Extrai a resposta do agente (primeira mensagem do tipo "assistant")
        reply = ""
        if hasattr(chat_resp, "messages") and chat_resp.messages:
            reply = chat_resp.messages[0].content
        if not reply:
            reply = "Desculpe, não consegui gerar uma resposta."
        return OpenAIChatResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(datetime.datetime.utcnow().timestamp()),
            model=getattr(req, "model", "sig9-agent-0.1"),
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=reply),
                    finish_reason="stop"
                )
            ]
        )
    except Exception as e:
        print(f"[ERROR] Erro inesperado no endpoint /v1/chat/completions: {e}")
        return OpenAIChatResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(datetime.datetime.utcnow().timestamp()),
            model=getattr(req, "model", "sig9-agent-0.1"),
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=f"Erro interno: {str(e)}"),
                    finish_reason="stop"
                )
            ]
        )

@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        rag_store.add_pdf(content, file.filename)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint for agent orchestration.
    Handles conversation state, agent routing, and guardrail checks.
    """
    try:
        # Initialize or retrieve conversation state
        is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
        if is_new:
            conversation_id: str = uuid4().hex
            ctx = create_initial_context()
            current_agent_name = triage_agent.name
            state: Dict[str, Any] = {
                "input_items": [],
                "context": ctx,
                "current_agent": current_agent_name,
            }
            if req.message.strip() == "":
                conversation_store.save(conversation_id, state)
                return ChatResponse(
                    conversation_id=conversation_id,
                    current_agent=current_agent_name,
                    messages=[],
                    events=[],
                    context=ctx.model_dump(),
                    agents=_build_agents_list(),
                    guardrails=[],
                )
        else:
            conversation_id = req.conversation_id  # type: ignore
            state = conversation_store.get(conversation_id)

        # --- INÍCIO: Lógica de menção de agente ---
        mentioned = extract_mention(req.message)
        if mentioned:
            agent_names = [a["name"] for a in _build_agents_list()]
            if mentioned in agent_names:
                current_agent = _get_agent_by_name(mentioned)
                state["current_agent"] = mentioned
                logger.info(f"Menção detectada: @{mentioned} - roteando para agente correspondente.")
            else:
                logger.warning(f"Menção a agente inexistente: @{mentioned}")
                return ChatResponse(
                    conversation_id=conversation_id,
                    current_agent=state["current_agent"],
                    messages=[MessageResponse(content=f"Agente @{mentioned} não encontrado.", agent="system")],
                    events=[],
                    context=state["context"].model_dump(),
                    agents=_build_agents_list(),
                    guardrails=[],
                )
        else:
            current_agent = _get_agent_by_name(state["current_agent"])
        # --- FIM: Lógica de menção de agente ---
        state["input_items"].append({"content": req.message, "role": "user"})
        old_context = state["context"].model_dump().copy()
        guardrail_checks: List[GuardrailCheck] = []

        try:
            result = await Runner.run(current_agent, state["input_items"], context=state["context"])
        except InputGuardrailTripwireTriggered as e:
            failed = e.guardrail_result.guardrail
            gr_output = e.guardrail_result.output.output_info
            gr_reasoning = getattr(gr_output, "reasoning", "")
            gr_input = req.message
            gr_timestamp = time.time() * 1000
            for g in current_agent.input_guardrails:
                guardrail_checks.append(GuardrailCheck(
                    id=uuid4().hex,
                    name=_get_guardrail_name(g),
                    input=gr_input,
                    reasoning=(gr_reasoning if g == failed else ""),
                    passed=(g != failed),
                    timestamp=gr_timestamp,
                ))
            refusal = "Sorry, I can only answer questions related to airline travel."
            state["input_items"].append({"role": "assistant", "content": refusal})
            return ChatResponse(
                conversation_id=conversation_id,
                current_agent=current_agent.name,
                messages=[MessageResponse(content=refusal, agent=current_agent.name)],
                events=[],
                context=state["context"].model_dump(),
                agents=_build_agents_list(),
                guardrails=guardrail_checks,
            )

        messages: List[MessageResponse] = []
        events: List[AgentEvent] = []

        for item in result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                messages.append(MessageResponse(content=text, agent=item.agent.name))
                events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))
            # Handle handoff output and agent switching
            elif isinstance(item, HandoffOutputItem):
                # Record the handoff event
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="handoff",
                        agent=item.source_agent.name,
                        content=f"{item.source_agent.name} -> {item.target_agent.name}",
                        metadata={"source_agent": item.source_agent.name, "target_agent": item.target_agent.name},
                    )
                )
                # If there is an on_handoff callback defined for this handoff, show it as a tool call
                from_agent = item.source_agent
                to_agent = item.target_agent
                # Find the Handoff object on the source agent matching the target
                ho = next(
                    (h for h in getattr(from_agent, "handoffs", [])
                     if isinstance(h, Handoff) and getattr(h, "agent_name", None) == to_agent.name),
                    None,
                )
                if ho:
                    fn = ho.on_invoke_handoff
                    fv = fn.__code__.co_freevars
                    cl = fn.__closure__ or []
                    if "on_handoff" in fv:
                        idx = fv.index("on_handoff")
                        if idx < len(cl) and cl[idx].cell_contents:
                            cb = cl[idx].cell_contents
                            cb_name = getattr(cb, "__name__", repr(cb))
                            events.append(
                                AgentEvent(
                                    id=uuid4().hex,
                                    type="tool_call",
                                    agent=to_agent.name,
                                    content=cb_name,
                                )
                            )
                current_agent = item.target_agent
            elif isinstance(item, ToolCallItem):
                tool_name = getattr(item.raw_item, "name", None)
                raw_args = getattr(item.raw_item, "arguments", None)
                tool_args: Any = raw_args
                if isinstance(raw_args, str):
                    try:
                        import json
                        tool_args = json.loads(raw_args)
                    except Exception:
                        pass
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="tool_call",
                        agent=item.agent.name,
                        content=tool_name or "",
                        metadata={"tool_args": tool_args},
                    )
                )
                # If the tool is display_seat_map, send a special message so the UI can render the seat selector.
                if tool_name == "display_seat_map":
                    messages.append(
                        MessageResponse(
                            content="DISPLAY_SEAT_MAP",
                            agent=item.agent.name,
                        )
                    )
            elif isinstance(item, ToolCallOutputItem):
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="tool_output",
                        agent=item.agent.name,
                        content=str(item.output),
                        metadata={"tool_result": item.output},
                    )
                )

        new_context = state["context"].dict()
        changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
        if changes:
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="context_update",
                    agent=current_agent.name,
                    content="",
                    metadata={"changes": changes},
                )
            )

        state["input_items"] = result.to_input_list()
        state["current_agent"] = current_agent.name
        conversation_store.save(conversation_id, state)

        # Build guardrail results: mark failures (if any), and any others as passed
        final_guardrails: List[GuardrailCheck] = []
        for g in getattr(current_agent, "input_guardrails", []):
            name = _get_guardrail_name(g)
            failed = next((gc for gc in guardrail_checks if gc.name == name), None)
            if failed:
                final_guardrails.append(failed)
            else:
                final_guardrails.append(GuardrailCheck(
                    id=uuid4().hex,
                    name=name,
                    input=req.message,
                    reasoning="",
                    passed=True,
                    timestamp=time.time() * 1000,
                ))

        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=messages,
            events=events,
            context=state["context"].dict(),
            agents=_build_agents_list(),
            guardrails=final_guardrails,
        )
    except Exception as e:
        import traceback
        logger.error("Erro no endpoint /chat: %s", traceback.format_exc())
        return ChatResponse(
            conversation_id="error",
            current_agent="error",
            messages=[MessageResponse(content=f"Erro interno: {str(e)}", agent="system")],
            events=[],
            context={},
            agents=_build_agents_list(),
            guardrails=[],
        )
