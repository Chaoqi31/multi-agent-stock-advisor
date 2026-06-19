"""
TechnicalAnalysis Agent: Performs technical analysis of a stock using ReAct Agent framework.
"""
import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.prebuilt import create_react_agent
import time

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(override=True)

logger = setup_logger(__name__)


async def technical_agent(state: AgentState) -> AgentState:
    """
    Perform technical analysis using the ReAct framework, directly integrating MCP tools

    Args:
        state: the current Agent state containing the user query

    Returns:
        The updated AgentState, including the technical analysis result
    """
    logger.info(f"{WAIT_ICON} TechnicalAgent: Starting technical analysis using ReAct framework.")

    # Get the execution logger, used to record the Agent's execution process
    execution_logger = get_execution_logger()
    agent_name = "technical_agent"

    # Extract the current data, messages, and metadata from the state
    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    user_query = current_data.get("query")

    # Record the start of the Agent's execution, including key information
    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "stock_code": current_data.get("stock_code"),
        "company_name": current_data.get("company_name"),
        "input_data_keys": list(current_data.keys())
    })

    # Validate that the user query exists
    if not user_query:
        logger.error(f"{ERROR_ICON} TechnicalAgent: User query is missing in state data.")
        current_data["technical_analysis_error"] = "User query is missing."
        execution_logger.log_agent_complete(agent_name, current_data, 0, False, "User query is missing")
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    # Record the Agent start time, used to compute the execution duration
    agent_start_time = time.time()

    try:
        # Use API calls
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

        # Validate that the required environment variables exist
        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} TechnicalAgent: Missing OpenAI environment variables.")
            current_data["technical_analysis_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing OpenAI environment variables")
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        logger.info(f"{WAIT_ICON} TechnicalAgent: Creating ChatOpenAI with model {model_name}")
        # Create the LLM instance, setting appropriate parameters
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,  # A lower temperature ensures consistency of the analysis
            max_tokens=6000   # Increase the number of tokens for detailed analysis
        )

        # 2. Fetch the MCP tool set
        logger.info(f"{WAIT_ICON} TechnicalAgent: Fetching MCP tools...")
        try:
            mcp_tools = await get_mcp_tools()
            if not mcp_tools:
                logger.error(f"{ERROR_ICON} TechnicalAgent: No MCP tools available.")
                current_data["technical_analysis_error"] = "No MCP tools available."
                execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "No MCP tools available")
                return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

            logger.info(f"{SUCCESS_ICON} TechnicalAgent: Successfully loaded {len(mcp_tools)} tools.")

            # Print the list of available tools, for ease of debugging
            tool_names = [tool.name for tool in mcp_tools]
            logger.info(f"Available tools: {tool_names}")

            # 3. Create the ReAct Agent - pass in only the LLM and tools
            logger.info(f"{WAIT_ICON} TechnicalAgent: Creating ReAct agent...")
            agent = create_react_agent(llm, mcp_tools)

            # 4. Prepare the input data, building a detailed analysis request
            stock_code = current_data.get('stock_code', 'Unknown')
            company_name = current_data.get('company_name', 'Unknown')
            current_time_info = current_data.get('current_time_info', '未知时间')
            current_date = current_data.get('current_date', '未知日期')

            # Build a detailed technical analysis request, covering multiple analysis dimensions
            agent_input = f"""Please analyze the technical indicators of {company_name} (stock ticker: {stock_code}).

Current time: {current_time_info}
Current date: {current_date}

Please perform the following technical analysis:
1. Obtain the stock's basic information and latest price
2. Obtain historical candlestick (K-line) data (it is recommended to obtain the most recent 3-6 months of data)
3. Analyze the price trend and technical patterns
4. Analyze volume changes
5. Compute and analyze the main technical indicators (such as moving averages, MACD, RSI, etc.)
6. Identify support and resistance levels
7. Provide a technical summary and a short-term trend judgment

Important constraint: Please focus on price data and technical indicator analysis; do not use the crawl_news tool to obtain news information. Technical analysis should be based on price action, volume, and technical indicator data, not on news events.

Please use the available tools to obtain actual data for the analysis, rather than relying on assumptions.

请用简体中文输出。"""

            logger.info(f"Agent input: {agent_input}")

            # 5. Call the ReAct Agent - using the correct messages format
            logger.info(f"{WAIT_ICON} TechnicalAgent: Calling ReAct agent...")
            start_time = time.time()

            # The LangGraph ReAct Agent requires input in the messages format
            input_data = {
                "messages": [HumanMessage(content=agent_input)]
            }

            # Call the Agent to perform the analysis
            response = await agent.ainvoke(input_data)

            end_time = time.time()
            execution_time = end_time - start_time

            logger.info(f"ReAct agent execution completed in {execution_time:.2f} seconds")

            # 6. Extract the analysis result
            final_output = "No analysis generated."

            if "messages" in response and isinstance(response["messages"], list):
                messages = response["messages"]
                # Find the last AI message, which usually contains the final analysis result
                ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    final_output = last_ai_message.content
                    logger.info(f"Successfully extracted analysis from AI message.")
                else:
                    logger.warning("No AI messages found in response")
                    # If there are no AI messages, try to obtain the content of all messages
                    all_content = []
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            all_content.append(str(msg.content))
                    if all_content:
                        final_output = "\n".join(all_content)
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                logger.error(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")

            logger.info(f"Final extracted analysis length: {len(final_output)} characters")
            print(f"TECHNICALAGENT: {final_output}")
            # 7. Log the LLM interaction, for subsequent analysis and optimization
            model_config = {
                "model": model_name,
                "temperature": 0.3,
                "max_tokens": 6000,
                "api_base": base_url
            }
            
            execution_logger.log_llm_interaction(
                agent_name=agent_name,
                interaction_type="react_agent",
                input_messages=[{"role": "user", "content": agent_input}],
                output_content=final_output,
                model_config=model_config,
                execution_time=execution_time
            )

            logger.info(f"{SUCCESS_ICON} TechnicalAgent: Successfully completed technical analysis.")

            # 8. Update the state, saving the analysis result and metadata
            current_data["technical_analysis"] = final_output
            current_metadata["technical_agent_executed"] = True
            current_metadata["technical_agent_timestamp"] = str(time.time())
            current_metadata["technical_agent_execution_time"] = f"{execution_time:.2f} seconds"

            # 9. Append a message record, preserving the conversation history
            new_message = {"role": "assistant", "content": "技术分析已完成"}
            updated_messages = current_messages + [new_message]

            # Record the Agent execution success
            total_execution_time = time.time() - agent_start_time
            execution_logger.log_agent_complete(agent_name, {
                "technical_analysis_length": len(final_output),
                "analysis_preview": final_output[:500] if len(final_output) > 500 else final_output,
                "llm_execution_time": execution_time,
                "total_execution_time": total_execution_time
            }, total_execution_time, True)

            return {
                "data": current_data,
                "messages": updated_messages,
                "metadata": current_metadata
            }

        except Exception as e:
            logger.error(f"{ERROR_ICON} TechnicalAgent: Error in MCP or agent execution: {e}", exc_info=True)
            current_data["technical_analysis_error"] = f"Error in MCP or agent execution: {e}"
            current_data["technical_analysis"] = f"技术分析过程中出现错误: {str(e)}"
            current_metadata["technical_agent_error"] = str(e)
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    except Exception as e:
        logger.error(f"{ERROR_ICON} TechnicalAgent: Error during execution: {e}", exc_info=True)
        current_data["technical_analysis_error"] = f"Error during execution: {e}"
        current_metadata["technical_agent_error"] = str(e)
        execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, str(e))
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}


# Local test function
async def test_technical_agent():
    """Test function for the technical analysis Agent"""
    from src.utils.state_definition import AgentState
    from datetime import datetime

    # Prepare test data, including the current time information
    current_datetime = datetime.now()
    current_date_cn = current_datetime.strftime("%Y年%m月%d日")
    current_date_en = current_datetime.strftime("%Y-%m-%d")
    current_weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][current_datetime.weekday()]
    current_time = current_datetime.strftime("%H:%M:%S")
    current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

    # Create the test state, simulating a real user query
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析 Apple (AAPL) 的技术指标",
            "stock_code": "AAPL",
            "company_name": "Apple",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        },
        metadata={}
    )

    # Run the Agent and output the result
    result = await technical_agent(test_state)
    print("Technical Analysis Result:")
    print(result.get("data", {}).get("technical_analysis", "No analysis found"))

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_technical_agent()) 