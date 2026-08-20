import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Dict, List, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("chatbot")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_config.json")


class MCPClientManager:
    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.tools_map: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Loads mcp_config.json and connects to all enabled MCP servers."""
        if not os.path.exists(self.config_path):
            logger.info(f"MCP config not found at {self.config_path}, skipping MCP server connections.")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return

        mcp_servers = config.get("mcpServers", {})
        for name, server_info in mcp_servers.items():
            if not server_info.get("enabled", True):
                logger.info(f"MCP server '{name}' is disabled in config.")
                continue

            command = server_info.get("command")
            args = server_info.get("args", [])
            env = server_info.get("env")

            if not command:
                logger.warning(f"MCP server '{name}' missing command, skipping.")
                continue

            actual_command = shutil.which(command) or shutil.which(command + ".cmd") or command
            server_env = os.environ.copy()
            if env:
                server_env.update(env)

            try:
                server_params = StdioServerParameters(command=actual_command, args=args, env=server_env)
                read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                self.sessions[name] = session
                logger.info(f"Successfully connected to MCP server '{name}'.")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{name}': {e}", exc_info=True)

    async def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Collects tools from all active MCP sessions and converts to OpenAI function schemas."""
        openai_tools = []
        new_tools_map: Dict[str, Dict[str, Any]] = {}

        for server_name, session in self.sessions.items():
            try:
                mcp_tools = await session.list_tools()
                for tool in mcp_tools.tools:
                    full_name = f"mcp__{server_name}__{tool.name}"
                    new_tools_map[full_name] = {
                        "server_name": server_name,
                        "original_name": tool.name
                    }

                    # Extract input schema from Pydantic Tool object (uses input_schema in Python MCP SDK)
                    if hasattr(tool, "input_schema") and tool.input_schema:
                        raw = tool.input_schema
                        schema = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
                    elif hasattr(tool, "inputSchema") and tool.inputSchema:
                        raw = tool.inputSchema
                        schema = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
                    else:
                        schema = {"type": "object", "properties": {}}

                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": full_name,
                            "description": tool.description or f"MCP tool '{tool.name}' from '{server_name}'",
                            "parameters": schema
                        }
                    })
            except Exception as e:
                logger.error(f"Failed to list tools from MCP server '{server_name}': {e}")

        self.tools_map.update(new_tools_map)
        return openai_tools

    async def call_tool(self, full_tool_name: str, arguments: dict) -> str:
        """Dispatches tool execution to the appropriate MCP server session."""
        info = self.tools_map.get(full_tool_name)
        if not info:
            return f"Error: MCP tool '{full_tool_name}' not registered."

        server_name = info["server_name"]
        original_name = info["original_name"]

        session = self.sessions.get(server_name)
        if not session:
            return f"Error: MCP server '{server_name}' session is not active."

        try:
            logger.info(f"Calling MCP tool '{original_name}' on server '{server_name}' with args: {arguments}")
            result = await session.call_tool(original_name, arguments)
            contents = []
            if hasattr(result, "content") and result.content:
                for item in result.content:
                    if hasattr(item, "text"):
                        contents.append(item.text)
                    else:
                        contents.append(str(item))
            output = "\n".join(contents) if contents else "Tool executed successfully (no text output)."
            logger.info(f"MCP tool '{full_tool_name}' executed successfully ({len(output)} chars).")
            return output
        except Exception as e:
            logger.error(f"MCP tool '{full_tool_name}' execution failed: {e}", exc_info=True)
            return f"MCP tool error: {e}"

    async def close(self):
        """Cleanly closes all active MCP client sessions."""
        try:
            await self.exit_stack.aclose()
            logger.info("Closed all MCP server connections.")
        except Exception as e:
            logger.error(f"Error closing MCP servers: {e}")
        self.sessions.clear()
        self.tools_map.clear()


# Global instance
mcp_manager = MCPClientManager()
