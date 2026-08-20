import datetime
import math
import os
import platform
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Tavily setup ──
try:
    from tavily import TavilyClient
    _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None
except ImportError:
    _tavily = None


# ── Tool functions ──

def get_current_time(timezone_name: str = "UTC") -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return f"Current time ({timezone_name}): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"


def calculator(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def system_info() -> str:
    return f"OS: {platform.system()} {platform.release()}, Python: {sys.version.split()[0]}"


def web_search(query: str, max_results: int = 5) -> str:
    if not _tavily:
        return "Web search unavailable. Check TAVILY_API_KEY in .env."
    try:
        response = _tavily.search(query=query, max_results=max_results)
        results = response.get("results", [])
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results, 1):
            title = r.get('title', '').encode('ascii', 'ignore').decode()
            url = r.get('url', '')
            content = r.get('content', '')[:300].encode('ascii', 'ignore').decode()
            output.append(f"{i}. {title}\n   {url}\n   {content}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


# ── Tool registry ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_name": {"type": "string", "default": "UTC", "description": "Timezone name e.g. UTC, EST"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression like '12 * 45'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "Retrieve system platform details.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current information, news, or any topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "default": 5, "description": "Number of results"}
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_MAP = {
    "get_current_time": lambda args: get_current_time(args.get("timezone_name", "UTC")),
    "calculator":       lambda args: calculator(args.get("expression", "")),
    "system_info":      lambda args: system_info(),
    "web_search":       lambda args: web_search(args.get("query", ""), args.get("max_results", 5)),
}


def call_tool(name: str, arguments: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Tool '{name}' not found."
    return fn(arguments)


async def get_all_tools() -> list:
    """Combines built-in tools with active MCP tools."""
    from connectors.mcp_client import mcp_manager
    mcp_tools = await mcp_manager.get_openai_tools()
    return TOOLS + mcp_tools


async def async_call_tool(name: str, arguments: dict) -> str:
    """Executes either a built-in tool or an MCP tool asynchronously."""
    if name.startswith("mcp__"):
        from connectors.mcp_client import mcp_manager
        return await mcp_manager.call_tool(name, arguments)
    return call_tool(name, arguments)

