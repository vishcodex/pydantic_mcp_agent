from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from dotenv import load_dotenv
import asyncio
import pathlib
import sys
import os

from pydantic_ai import Agent
from openai import AsyncOpenAI, OpenAI
from pydantic_ai.models.openai import OpenAIModel

import mcp_client

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Define the path to the config file relative to the script directory
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"

load_dotenv()

# Configure and return the pydantic-ai model wrapper with OpenRouter support
def get_model():
    llm = os.getenv('MODEL_CHOICE', 'anthropic/claude-3.5-sonnet')
    base_url = os.getenv('BASE_URL', 'https://openrouter.ai/api/v1')
    api_key = os.getenv('OPENROUTER_API_KEY', os.getenv('LLM_API_KEY', 'no-api-key-provided'))

    # For OpenRouter, we need to set specific headers
    # Since OpenAIModel doesn't accept a client parameter, we'll use extra_headers
    # extra_headers = {
    #     "HTTP-Referer": "http://localhost",  # Required by OpenRouter
    #     "X-Title": "Pydantic MCP Agent"      # Required by OpenRouter
    # }

    # Instantiate the pydantic-ai wrapper
    model = OpenAIModel(
        llm,  # Positional model name
        base_url=base_url,
        api_key=api_key
    )
    
    # # Try to set headers on the underlying client if it exists
    # if hasattr(model, 'client') and model.client:
    #     model.client.default_headers.update(extra_headers)
    
    return model

# Reverted to original structure: get model instance and pass to Agent
async def get_pydantic_ai_agent():
    # Start MCP client for tools
    mcp_instance = mcp_client.MCPClient() # Renamed local variable
    mcp_instance.load_servers(str(CONFIG_FILE))
    tools = await mcp_instance.start()

    # Instantiate Agent using the OpenAIModel instance
    agent = Agent(model=get_model(), tools=tools) # Pass the model instance

    return mcp_instance, agent # Return renamed variable

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # ~~~~~~~~~~~~~~~~~~~~ Main Function with CLI Chat ~~~~~~~~~~~~~~~~~~~~~
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    print("=== Pydantic AI MCP CLI Chat ===")
    print("Type 'exit' to quit the chat")
    
    # Initialize the agent and message history
    mcp_instance, mcp_agent = await get_pydantic_ai_agent() # Use renamed variable
    # print(mcp_instance, mcp_agent) # Removed debug print
    console = Console()
    messages = []
    
    try:
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
            
            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                with Live('', console=console, vertical_overflow='visible') as live:
                    async with mcp_agent.run_stream(
                        user_input, message_history=messages
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))
                    
                    # Add the new messages to the chat history
                    messages.extend(result.all_messages())
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
    finally:
        # Ensure proper cleanup of MCP client resources when exiting
        await mcp_instance.cleanup()

if __name__ == "__main__":
    asyncio.run(main())