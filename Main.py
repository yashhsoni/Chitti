import asyncio
import json
import os
import sys
import time
import logging
import re
from collections import defaultdict

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from fastapi.templating import Jinja2Templates
from connectors.tools import get_all_tools, async_call_tool
from connectors.mcp_client import mcp_manager
from db import init_db, save_message, load_history, get_all_sessions, save_memory, load_memories, delete_memory, delete_session

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("chatbot")

MAX_MESSAGE_LENGTH = 1000
RATE_LIMIT_MESSAGES = 10
RATE_LIMIT_WINDOW = 60
_rate_tracker: dict = defaultdict(list)

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"disregard (all |previous |above )?instructions",
    r"you are now",
    r"act as (a |an )?(different|new|another)?",
    r"forget (everything|all|your instructions|your training)",
    r"new persona",
    r"jailbreak",
    r"do anything now",
    r"dan mode",
]

def is_rate_limited(client_id: str) -> bool:
    now = time.time()
    _rate_tracker[client_id] = [t for t in _rate_tracker[client_id] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_tracker[client_id]) >= RATE_LIMIT_MESSAGES:
        return True
    _rate_tracker[client_id].append(now)
    return False

def has_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in PROMPT_INJECTION_PATTERNS)

chat_client = OpenAI(
    base_url="https://openai.generative.engine.capgemini.com/v1",
    api_key=os.getenv("CHAT_API_KEY")
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing MCP Client Manager...")
    await mcp_manager.initialize()
    yield
    logger.info("Closing MCP Client Manager...")
    await mcp_manager.close()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")
init_db()


@app.get("/sessions")
async def get_sessions():
    return get_all_sessions()


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    delete_session(session_id)
    return {"ok": True}


@app.get("/memories")
async def get_memories():
    return load_memories()


@app.delete("/memories/{memory_id}")
async def remove_memory(memory_id: int):
    delete_memory(memory_id)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "home.html", {})


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    client_id = str(websocket.client.host)
    session_id = None
    chat_log = []
    logger.info(f"WebSocket connected: {client_id}")

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Handle session init — load history and send it back to UI
            if message_data.get("action") == "init":
                session_id = message_data.get("session_id", f"{client_id}-{int(time.time())}")
                chat_log = load_history(session_id)
                history_to_send = [m for m in chat_log if m["role"] != "system"]
                logger.info(f"Session init: {session_id} ({len(history_to_send)} messages loaded)")
                await websocket.send_json({"type": "history", "data": history_to_send})
                continue

            # Handle load existing session
            if message_data.get("action") == "load_session":
                session_id = message_data.get("session_id")
                chat_log = load_history(session_id)
                history_to_send = [m for m in chat_log if m["role"] != "system"]
                await websocket.send_json({"type": "history", "data": history_to_send})
                continue

            # Handle new chat
            if message_data.get("action") == "new_chat":
                session_id = f"{client_id}-{int(time.time())}"
                chat_log = load_history(session_id)
                continue

            # Fallback if init was never received
            if not session_id:
                session_id = f"{client_id}-{int(time.time())}"
                chat_log = load_history(session_id)

            user_input = message_data.get("message", "").strip()

            if not user_input:
                await websocket.send_json({"type": "error", "data": "Message cannot be empty"})
                continue

            if len(user_input) > MAX_MESSAGE_LENGTH:
                await websocket.send_json({"type": "error", "data": f"Message too long. Max {MAX_MESSAGE_LENGTH} characters."})
                continue

            if is_rate_limited(client_id):
                await websocket.send_json({"type": "error", "data": "Too many messages. Please slow down."})
                continue

            if has_prompt_injection(user_input):
                await websocket.send_json({"type": "error", "data": "Your message was flagged as a potential prompt injection attempt."})
                continue

            # Detect and save memory requests
            remember_match = re.search(r"(?:please )?remember\s+(?:that\s+)?(.+)", user_input, re.IGNORECASE)
            if remember_match:
                save_memory(remember_match.group(1).strip())
                chat_log = load_history(session_id)

            # Detect forget requests — delete matching memories
            forget_match = re.search(r"(?:please )?forget\s+(?:that\s+)?(.+)", user_input, re.IGNORECASE)
            if forget_match:
                keyword = forget_match.group(1).strip().lower()
                for m in load_memories():
                    if keyword in m["memory"].lower():
                        delete_memory(m["id"])
                chat_log = load_history(session_id)

            logger.info(f"User: '{user_input}'")
            chat_log.append({"role": "user", "content": user_input})
            save_message(session_id, "user", user_input)
            await websocket.send_json({"type": "typing_start"})

            try:
                active_tools = await get_all_tools()
                kwargs = {
                    "model": "anthropic.claude-sonnet-4-6",
                    "messages": chat_log,
                    "temperature": 0.3,
                }
                if active_tools:
                    kwargs["tools"] = active_tools

                current_choice = chat_client.chat.completions.create(**kwargs, stream=False).choices[0]
                bot_response = ""

                # Agentic loop: keep handling tool calls until model returns text
                while current_choice.message.tool_calls:
                    chat_log.append(current_choice.message.model_dump(exclude_unset=True, exclude_none=True))

                    for tool_call in current_choice.message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                        except Exception:
                            func_args = {}

                        await websocket.send_json({"type": "tool_call", "data": f"Running: {func_name}..."})
                        tool_result = await async_call_tool(func_name, func_args)
                        chat_log.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result)})

                    current_choice = chat_client.chat.completions.create(
                        model="anthropic.claude-sonnet-4-6",
                        messages=chat_log,
                        tools=active_tools,
                        temperature=0.6,
                        stream=False
                    ).choices[0]

                bot_response = current_choice.message.content or ""
                if bot_response:
                    await websocket.send_json({"type": "chunk", "data": bot_response})

                chat_log.append({"role": "assistant", "content": bot_response})
                save_message(session_id, "assistant", bot_response)
                logger.info(f"Bot response complete ({len(bot_response)} chars)")
                await websocket.send_json({"type": "done"})

            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "data": str(e)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Main:app", host="0.0.0.0", port=8000, reload=True)
