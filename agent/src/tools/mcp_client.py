from langchain_mcp_adapters.client import MultiServerMCPClient
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.tools.mcp_config import SERVER_CONFIGS
import asyncio  # Required for async operations, such as get_tools
import json

logger = setup_logger(__name__)

_mcp_client_instance = None
_mcp_tools = None


def print_tool_details(tools):
    """Print detailed information about the tools, for debugging."""
    logger.info(f"{SUCCESS_ICON} Tool details:")
    for i, tool in enumerate(tools, 1):
        logger.info(f"  {i}. Tool name: {tool.name}")
        logger.info(f"     Description: {tool.description}")

        # Print other possible attributes
        for attr in ['input_schema', 'parameters', 'schema']:
            if hasattr(tool, attr):
                attr_value = getattr(tool, attr)
                if attr_value:
                    logger.info(f"     {attr}: {attr_value}")

        logger.info(f"     Tool type: {type(tool)}")
        # logger.info(f"     All attributes: {dir(tool)}")
        logger.info("     " + "-" * 50)


async def get_mcp_tools():
    """
    Initialize the MultiServerMCPClient using the defined server configuration,
    and fetch the available tools from the us-stock-mcp-v2 server.

    Returns:
        list: a list of LangChain-compatible tools loaded from the MCP server.
              Returns an empty list if initialization or tool loading fails.
    """
    global _mcp_client_instance, _mcp_tools

    if _mcp_tools is not None:
        logger.info(f"{SUCCESS_ICON} Returning cached MCP tools.")
        return _mcp_tools

    logger.info(
        f"{WAIT_ICON} Initializing MultiServerMCPClient with config: {SERVER_CONFIGS}")
    try:
        _mcp_client_instance = MultiServerMCPClient(SERVER_CONFIGS)

        logger.info(
            f"{WAIT_ICON} Fetching tools from MCP server 'us_stock_mcp_v2'...")
        # The get_tools() method is asynchronous.
        loaded_tools = await _mcp_client_instance.get_tools()

        if not loaded_tools:
            logger.warning(
                f"{ERROR_ICON} No tools loaded from MCP server 'us_stock_mcp_v2'. Check server logs and configuration.")
            _mcp_tools = []  # Cache empty list on failure to load
            return []

        _mcp_tools = loaded_tools
        logger.info(
            f"{SUCCESS_ICON} Successfully loaded {len(_mcp_tools)} tools from 'us_stock_mcp_v2'.")

        # # Print the list of tool names
        # tool_names = [tool.name for tool in _mcp_tools]
        # logger.info(f"Tool name list: {tool_names}")

        # Print detailed tool information
        # print_tool_details(_mcp_tools)

        return _mcp_tools

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} Failed to initialize MCP client or load tools: {e}", exc_info=True)
        _mcp_tools = []  # Cache empty list on failure
        return []


async def close_mcp_client_sessions():
    """
    Close any open sessions managed by the MultiServerMCPClient.
    This function should be called on application shutdown if necessary.
    """
    global _mcp_client_instance
    if _mcp_client_instance:
        logger.info(f"{WAIT_ICON} Closing MCP client sessions...")
        try:
            logger.info(
                f"{SUCCESS_ICON} MCP client sessions (if any were persistently open) assumed closed or managed by library.")
            _mcp_client_instance = None   # Allow re-initialization
            global _mcp_tools
            _mcp_tools = None
        except Exception as e:
            logger.error(
                f"{ERROR_ICON} Error during MCP client session cleanup: {e}", exc_info=True)
    else:
        logger.info("MCP client was not initialized, no sessions to close.")


# Example for testing this module (optional, for direct execution)
async def _main_test_mcp_client():
    logger.info("--- Testing MCP Client Tool Loading ---")
    tools = await get_mcp_tools()
    if tools:
        print(f"Successfully loaded {len(tools)} tools:")
        for tool in tools:
            print(
                f"- Name: {tool.name}")

        # Test a simple tool call (if a suitable tool is available)
        if tools:
            logger.info("--- Testing Tool Call ---")
            # Try calling the first tool (parameters need to be adjusted to the actual tool)
            first_tool = tools[0]
            logger.info(f"Attempting to call tool: {first_tool.name}")

            # Test parameters need to be constructed based on the actual tool parameter schema
            # Skip the actual call for now, just demonstrate the structure
            logger.info("Tool call test skipped (requires actual parameters)")
    else:
        print("Failed to load tools or no tools found.")

    # Test the shutdown (if applicable)
    await close_mcp_client_sessions()
    logger.info("--- MCP Client Test Complete ---")

if __name__ == '__main__':
    # This allows running the test directly, e.g.: python -m src.tools.mcp_client
    # Make sure your environment is set up (e.g., the 'uv' command is available).

    # Set up basic logging for the test run if not already configured
    if not logger.hasHandlers():
        import logging
        logging.basicConfig(level=logging.INFO)
        logger.info("Basic logging configured for test run.")

    asyncio.run(_main_test_mcp_client())
