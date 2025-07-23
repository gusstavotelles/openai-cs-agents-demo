import os
import asyncio
import aiohttp
import socketio
import json
import time
from utils import similarity, acquire_lock, release_lock, CFG, logger

WEBUI = os.getenv("WEBUI_URL", "http://open-webui:8080")
TOKEN  = os.getenv("TOKEN", "")
AGENT_API = os.getenv("AGENT_API", "http://agent-hub:8000/v1/chat/completions/stream")

sio = socketio.AsyncClient(logger=False)

def should_answer(text):
    if len(text) < 10 or "http" in text:
        return None
    for name, cfg in CFG.items():
        if name == "global":
            continue
        if similarity(text, cfg["keywords"]) >= CFG["global"]["similarity_threshold"]:
            return name
    return None

async def post_msg(chan, content):
    url = f"{WEBUI}/api/v1/channels/{chan}/messages/post"
    hdr = {"Authorization": f"Bearer {TOKEN}"}
    async with aiohttp.ClientSession() as s:
        await s.post(url, headers=hdr, json={"content": content})

async def call_agent(history):
    async with aiohttp.ClientSession() as s:
        async with s.post(AGENT_API, json={
            "model": "sig9-agent-0.1",
            "messages": history,
            "stream": True
        }) as resp:
            buf = ""
            async for line in resp.content:
                if line.startswith(b"data: "):
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    buf += chunk["choices"][0]["delta"].get("content", "")
            return buf.strip()

def history_from(evt):
    return [{"role": "user", "content": evt["data"]["data"]["content"]}]

@sio.on("channel-events")
async def on_event(evt):
    if evt["data"]["type"] != "message" or evt["user"].get("is_bot", False):
        return
    text = evt["data"]["data"]["content"]
    agent = should_answer(text)
    if not agent:
        return
    chan = evt["channel_id"]
    if not await acquire_lock(chan):
        return
    await sio.emit("channel-events", {
        "channel_id": chan,
        "data": {"type": "typing", "data": {"typing": True}}
    })
    answer = await call_agent(history_from(evt))
    await post_msg(chan, answer)
    await release_lock(chan)

async def main():
    await sio.connect(
        WEBUI,
        socketio_path="/ws/socket.io",
        headers={"Authorization": f"Bearer {TOKEN}"}
    )
    await sio.wait()

if __name__ == "__main__":
    asyncio.run(main())
