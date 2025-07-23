"""
Bot Worker para integração com OpenWebUI Channels via Socket.IO.

- Conecta ao servidor WebSocket do OpenWebUI (ex: ws://localhost:8080/ws/socket.io).
- Escuta mensagens em canais.
- Detecta menções (@agent) e, ao ser mencionado, consulta a API local FastAPI para processar a mensagem com o agente correspondente.
- Posta a resposta no canal, suportando "typing..." e resposta final.
- Pode ser rodado como processo separado, mas depende da API local estar rodando.

Requisitos:
    pip install python-socketio requests

Configuração:
    - Defina as variáveis abaixo conforme seu ambiente.
    - O bot precisa de um API Key de usuário-bot criado no OpenWebUI.
"""

import os
import re
import socketio
import requests
import threading
from typing import List, Dict, Any, Optional

# Configurações
OPENWEBUI_WS_URL = os.environ.get("OPENWEBUI_WS_URL", "ws://localhost:8080/ws/socket.io/?EIO=4&transport=websocket")
OPENWEBUI_API_URL = os.environ.get("OPENWEBUI_API_URL", "http://localhost:8080/api")
BOT_API_KEY = os.environ.get("BOT_API_KEY", "")
BOT_NAME = os.environ.get("BOT_NAME", "sig9-bot")
LOCAL_AGENT_API_URL = os.environ.get("LOCAL_AGENT_API_URL", "http://localhost:8000/chat")

MENTION_REGEX = re.compile(r'@([a-zA-Z0-9_]+)')
sio = socketio.Client(reconnection=True, reconnection_attempts=5, reconnection_delay=2)

def get_channel_history(channel_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Busca o histórico recente do canal via API REST do OpenWebUI."""
    headers = {"Authorization": f"Bearer {BOT_API_KEY}"}
    url = f"{OPENWEBUI_API_URL}/channels/{channel_id}/messages?limit={limit}"
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.ok:
            return resp.json().get("messages", [])
    except Exception as e:
        print(f"[BotWorker] Erro ao buscar histórico do canal: {e}")
    return []

def send_typing(channel_id: str) -> None:
    """Envia evento de 'typing...' para o canal."""
    try:
        sio.emit("typing", {"channel": channel_id, "user": BOT_NAME})
    except Exception as e:
        print(f"[BotWorker] Erro ao enviar typing: {e}")

def send_message(channel_id: str, content: str) -> bool:
    """Envia mensagem para o canal."""
    headers = {"Authorization": f"Bearer {BOT_API_KEY}"}
    url = f"{OPENWEBUI_API_URL}/channels/{channel_id}/messages"
    data = {"content": content}
    try:
        resp = requests.post(url, json=data, headers=headers, timeout=10)
        return resp.ok
    except Exception as e:
        print(f"[BotWorker] Erro ao enviar mensagem: {e}")
        return False

def process_mention(agent_name: str, message: str, channel_id: str, user: str, history: List[Dict[str, Any]]) -> None:
    """Chama a API local para processar a menção e posta a resposta no canal."""
    send_typing(channel_id)
    try:
        resp = requests.post(
            LOCAL_AGENT_API_URL,
            json={"conversation_id": channel_id, "message": message},
            timeout=30
        )
        reply = None
        if resp.ok:
            data = resp.json()
            reply = next((m["content"] for m in data.get("messages", []) if m.get("content")), None)
        if not reply:
            reply = "Desculpe, não consegui gerar uma resposta."
    except Exception as e:
        reply = f"Erro ao consultar agente: {e}"
    send_message(channel_id, reply)

@sio.event
def connect():
    print("[BotWorker] Conectado ao OpenWebUI WebSocket.")

@sio.event
def disconnect():
    print("[BotWorker] Desconectado do WebSocket.")

@sio.on("message")
def on_message(data: Dict[str, Any]):
    """
    Evento disparado ao receber mensagem em um canal.
    data: {
        "channel": "id_do_canal",
        "content": "texto",
        "author": {"name": "user", ...}
    }
    """
    content = data.get("content", "")
    channel_id = data.get("channel")
    user = data.get("author", {}).get("name", "")
    if not content or not channel_id or user.lower() == BOT_NAME.lower():
        return  # Ignora mensagens do próprio bot

    match = MENTION_REGEX.search(content)
    if match and match.group(1).lower() == BOT_NAME.lower():
        print(f"[BotWorker] Menção detectada em canal {channel_id} por {user}: {content}")
        history = get_channel_history(channel_id)
        threading.Thread(
            target=process_mention,
            args=(match.group(1), content, channel_id, user, history),
            daemon=True
        ).start()

def main():
    if not BOT_API_KEY:
        print("[BotWorker] BOT_API_KEY não configurado. Configure a variável de ambiente.")
        return
    print(f"[BotWorker] Iniciando bot {BOT_NAME}...")
    try:
        sio.connect(OPENWEBUI_WS_URL, headers={"Authorization": f"Bearer {BOT_API_KEY}"})
        sio.wait()
    except Exception as e:
        print(f"[BotWorker] Falha ao conectar: {e}")

if __name__ == "__main__":
    main()
