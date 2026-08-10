import asyncio
import json
import os
import sys
import time
import logging
import re
from collections import defaultdict

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI
from fastapi.templating import Jinja2Templates

from connectors.tools import TOOLS, call_tool

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("chatbot")

# ── Guardrail Config ──
MAX_MESSAGE_LENGTH = 1000          # max characters per message
RATE_LIMIT_MESSAGES = 10           # max messages per window
RATE_LIMIT_WINDOW = 60             # seconds
_rate_tracker: dict = defaultdict(list)

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"disregard (all |previous |above )?instructions",
    r"you are now",
    r"act as (a |an )?(different|new|another)?",
    r"forget (everything|all|your instructions)",
    r"new persona",
    r"jailbreak",
    r"do anything now",
    r"dan mode",
]

def is_rate_limited(client_id: str) -> bool:
    now = time.time()
    timestamps = _rate_tracker[client_id]
    _rate_tracker[client_id] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_tracker[client_id]) >= RATE_LIMIT_MESSAGES:
        return True
    _rate_tracker[client_id].append(now)
    return False

def has_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in PROMPT_INJECTION_PATTERNS)

# Chat client - uses OpenAI-compatible endpoint
chat_client = OpenAI(
    base_url="https://openai.generative.engine.capgemini.com/v1",
    api_key=os.getenv("CHAT_API_KEY")
)

app = FastAPI()

templates = Jinja2Templates(directory="template")
chat_responses = []

chat_log = [{'role': 'system', 'content': 'You are a helpful assistant. You can use available connectors/tools to answer user requests.'}]


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    logger.info("Serving home chat page")
    return templates.TemplateResponse(request, "home.html", {"chat_responses": chat_responses})


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_input = message_data.get("message", "").strip()

            if not user_input:
                logger.warning("Empty message received from client")
                await websocket.send_json({"type": "error", "data": "Message cannot be empty"})
                continue

            # Guardrail: message length
            if len(user_input) > MAX_MESSAGE_LENGTH:
                logger.warning(f"Message too long: {len(user_input)} chars")
                await websocket.send_json({"type": "error", "data": f"Message too long. Max {MAX_MESSAGE_LENGTH} characters allowed."})
                continue

            # Guardrail: rate limiting
            client_id = str(websocket.client)
            if is_rate_limited(client_id):
                logger.warning(f"Rate limit exceeded for {client_id}")
                await websocket.send_json({"type": "error", "data": "Too many messages. Please slow down."})
                continue

            # Guardrail: prompt injection
            if has_prompt_injection(user_input):
                logger.warning(f"Prompt injection attempt detected: '{user_input}'")
                await websocket.send_json({"type": "error", "data": "Your message was flagged as a potential prompt injection attempt."})
                continue

            logger.info(f"User message: '{user_input}'")
            chat_log.append({'role': 'user', 'content': user_input})
            chat_responses.append(user_input)
            await websocket.send_json({"type": "typing_start"})

            try:
                tools = TOOLS
                logger.info(f"Available tools: {[t['function']['name'] for t in tools]}")

                kwargs = {
                    "model": 'anthropic.claude-sonnet-4-6',
                    "messages": chat_log,
                    "temperature": 0.3,
                }
                if tools:
                    kwargs["tools"] = tools

                first_resp = chat_client.chat.completions.create(**kwargs, stream=False)
                choice = first_resp.choices[0]

                if choice.message.tool_calls:
                    assistant_msg = choice.message
                    chat_log.append(assistant_msg)

                    for tool_call in choice.message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                        except Exception:
                            func_args = {}

                        logger.info(f"LLM requested tool call '{func_name}' with args: {func_args}")
                        await websocket.send_json({"type": "tool_call", "data": f"⚡ Running connector: {func_name}..."})

                        tool_result = await asyncio.get_event_loop().run_in_executor(
                            None, lambda fn=func_name, fa=func_args: call_tool(fn, fa)
                        )
                        logger.info(f"Tool '{func_name}' output: {tool_result}")

                        chat_log.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result)
                        })

                    # Stream final response after tool execution
                    stream = chat_client.chat.completions.create(
                        model='anthropic.claude-sonnet-4-6',
                        messages=chat_log,
                        temperature=0.6,
                        stream=True
                    )
                    bot_response = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            bot_response += content
                            await websocket.send_json({"type": "chunk", "data": content})
                            await asyncio.sleep(0.01)

                    chat_log.append({'role': 'assistant', 'content': bot_response})
                    chat_responses.append(bot_response)
                    logger.info(f"Bot response complete ({len(bot_response)} characters)")
                    await websocket.send_json({"type": "done"})

                else:
                    bot_response = choice.message.content or ""
                    if bot_response:
                        await websocket.send_json({"type": "chunk", "data": bot_response})
                    chat_log.append({'role': 'assistant', 'content': bot_response})
                    chat_responses.append(bot_response)
                    logger.info(f"Bot response complete ({len(bot_response)} characters)")
                    await websocket.send_json({"type": "done"})

            except Exception as e:
                logger.error(f"Chat completion error: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "data": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "Main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
        reload_excludes=["logs/*", "*.log"]
    )