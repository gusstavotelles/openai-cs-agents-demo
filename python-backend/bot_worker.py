"""
Bot Worker para integração OpenWebUI + Agents API (sig9.dev)
------------------------------------------------------------
- Escuta eventos de canal do OpenWebUI via websocket oficial (Socket.IO)
- Envia mensagens de humanos para a API customizada de agents
- Posta a resposta como bot no canal

Pré-requisitos:
- Python 3.11+
- pip install -r requirements.txt
- Gere uma API-key para o usuário-bot no painel do OpenWebUI
- Adicione o usuário-bot ao(s) canal(is) desejado(s)

Configuração:
- Defina as variáveis de ambiente:
    WEBUI_URL: URL do OpenWebUI (ex: http://localhost:8080)
    BOT_TOKEN: API-key do usuário-bot
    AGENT_API: Endpoint da sua API customizada (ex: http://localhost:8000/v1/chat/completions)

Execução:
    export WEBUI_URL=http://localhost:8080
    export BOT_TOKEN=coloque_sua_api_key_aqui
    export AGENT_API=http://localhost:8000/v1/chat/completions
    python bot_worker.py
"""

import os
import asyncio
import aiohttp
import socketio
import time
from collections import defaultdict

WEBUI  = os.getenv("WEBUI_URL",  "http://localhost:8080")
TOKEN  = os.getenv("BOT_TOKEN")
AGENT  = os.getenv("AGENT_API", "http://localhost:8000/v1/chat/completions")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "5"))  # 5s por padrão

sio    = socketio.AsyncClient(reconnection=True, reconnection_attempts=999, reconnection_delay=2)
last_response = defaultdict(lambda: 0)  # (canal, usuário) -> timestamp

async def post_message(channel_id, content):
    url = f"{WEBUI}/api/v1/channels/{channel_id}/messages/post"
    async with aiohttp.ClientSession() as s:
        async with s.post(url,
                          headers={"Authorization": f"Bearer {TOKEN}"},
                          json={"content": content}) as resp:
            if resp.status != 200:
                print(f"[ERRO] Falha ao postar mensagem no canal {channel_id}: {resp.status} {await resp.text()}")

@sio.on("channel-events")
async def handle(evt):
    try:
        if evt["data"]["type"] != "message" or evt["user"]["is_bot"]:
            return

        user_id = evt["user"]["id"]
        channel = evt["channel_id"]
        user_msg = evt["data"]["data"]["content"]

        now = time.time()
        key = (channel, user_id)
        if now - last_response[key] < COOLDOWN_SECONDS:
            print(f"[COOLDOWN] Ignorando mensagem de {user_id} no canal {channel} (aguardando cooldown)")
            return
        last_response[key] = now

        print(f"[RECEBIDO] {user_id} em {channel}: {user_msg}")

        async with aiohttp.ClientSession() as s:
            resp = await s.post(AGENT, json={
                "model": "sig9-agent-0.1",
                "messages": [{"role": "user", "content": user_msg}]
            })
            if resp.status != 200:
                print(f"[ERRO] Falha ao chamar AGENT_API: {resp.status} {await resp.text()}")
                return
            data = await resp.json()
            reply = data["choices"][0]["message"]["content"]

        print(f"[ENVIANDO] Resposta para {channel}: {reply}")
        await post_message(channel, reply)
    except Exception as e:
        print(f"[EXCEPTION] {e}")

async def main():
    while True:
        try:
            print(f"Conectando ao OpenWebUI em {WEBUI} como bot...")
            await sio.connect(
                WEBUI,
                socketio_path="/ws/socket.io",
                headers={"Authorization": f"Bearer {TOKEN}"}
            )
            print("Conectado! Escutando eventos de canal...")
            await sio.wait()
        except Exception as e:
            print(f"[RECONNECT] Erro na conexão websocket: {e}. Tentando reconectar em 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
