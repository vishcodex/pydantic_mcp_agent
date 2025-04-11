from mcp_client import MCPClient
import pathlib
import asyncio

CONFIG_FILE = pathlib.Path('mcp_config.json')

async def main():
    client = MCPClient()
    client.load_servers(str(CONFIG_FILE))
    tools = await client.start()
    print("\nAvailable Tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
        if hasattr(tool, 'inputSchema'):
            print(f"  Input Schema: {tool.inputSchema}")

asyncio.run(main())
