import asyncio
import sqlite3
import sys
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_history.db")

app = Server("sqlite")


@app.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="execute_query",
            description="Execute a read-only SQL SELECT query on the chat history database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query to execute"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="list_tables",
            description="List all tables in the database with their schemas.",
            inputSchema={"type": "object", "properties": {}}
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "list_tables":
        with sqlite3.connect(DB_PATH) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            result = []
            for (table,) in tables:
                schema = conn.execute(f"PRAGMA table_info({table})").fetchall()
                cols = ", ".join(f"{col[1]} {col[2]}" for col in schema)
                result.append(f"{table}({cols})")
        return [types.TextContent(type="text", text="\n".join(result))]

    if name == "execute_query":
        query = arguments.get("query", "").strip()
        if not query.lower().startswith("select"):
            return [types.TextContent(type="text", text="Error: Only SELECT queries are allowed.")]
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query).fetchall()
                if not rows:
                    return [types.TextContent(type="text", text="No results found.")]
                headers = rows[0].keys()
                lines = [" | ".join(headers)]
                lines += [" | ".join(str(row[h]) for h in headers) for row in rows]
            return [types.TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Query error: {e}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
