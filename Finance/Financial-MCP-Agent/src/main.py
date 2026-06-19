"""
Financial Analysis AI Agent System Main Program

This file is the core entry point of the financial analysis AI agent system, implementing the following main features:

1. Multi-agent workflow management: build a parallel-executing agent workflow using LangGraph
2. Command-line interface: provide a user-friendly interactive command-line interface
3. Natural language processing: automatically identify and extract the stock ticker and company name
4. Logging system: complete execution logging and error handling
5. Report generation: generate a comprehensive financial analysis report

Workflow:
start_node -> [fundamental_analyst, technical_analyst, value_analyst, news_analyst] -> summarizer -> END
"""

# ============================================================================
# Import the necessary modules and dependencies
# ============================================================================

# Set environment variables before importing other modules, to suppress unhelpful output
import os
import sys

# Ensure the project root (the directory that contains the `src` package) is on the
# import path before any `from src...` imports run, regardless of how the script is
# launched (e.g. `python src/main.py` puts `src/` on the path, not the project root).
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Set environment variables to suppress redundant output from transformers and other libraries
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Only show error messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable the tokenizer parallelism warning
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Reduce CUDA-related output
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Reduce memory allocation messages

# Set the log level to suppress INFO-level output from third-party libraries
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Logging and state-management related imports
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.utils.state_definition import AgentState
from src.utils.execution_logger import initialize_execution_logger, finalize_execution_logger, get_execution_logger
from src.utils.stock_identifier import extract_stock_info as extract_us_stock_info, normalize_stock_code

# Agent module imports - the five core analysis agents
from src.agents.summary_agent import summary_agent      # Summary agent: consolidates all analysis results
from src.agents.value_agent import value_agent          # Valuation agent: analyzes the stock's valuation level
from src.agents.technical_agent import technical_agent  # Technical analysis agent: analyzes price trends and technical indicators
from src.agents.fundamental_agent import fundamental_agent  # Fundamental agent: analyzes financial condition and profitability
from src.agents.news_agent import news_agent            # News analysis agent: analyzes news sentiment and risk

# LangGraph workflow framework imports
from langgraph.graph import StateGraph, END

# Environment variable and system related imports
from dotenv import load_dotenv
import argparse
import asyncio
from datetime import datetime

# ============================================================================
# Initialization and configuration
# ============================================================================

# Set up the logger
logger = setup_logger(__name__)

# Load environment variables (from the .env file)
load_dotenv(override=True)

# Debug: print key environment variables to verify the configuration
logger.info(f"Environment Variables Loaded:")
logger.info(
    f"  OPENAI_COMPATIBLE_MODEL: {os.getenv('OPENAI_COMPATIBLE_MODEL', 'Not Set')}")
logger.info(
    f"  OPENAI_COMPATIBLE_BASE_URL: {os.getenv('OPENAI_COMPATIBLE_BASE_URL', 'Not Set')}")
logger.info(
    f"  OPENAI_COMPATIBLE_API_KEY: {'*' * 20 if os.getenv('OPENAI_COMPATIBLE_API_KEY') else 'Not Set'}")

# Re-initialize the logger (to ensure correct configuration)
logger = setup_logger(__name__)


async def main():
    """
    Main function: the core execution logic of the financial analysis AI agent system

    Features include:
    1. Initialize the execution logging system
    2. Build the LangGraph workflow
    3. Handle command-line arguments and user input
    4. Extract stock information (ticker, company name)
    5. Execute the multi-agent analysis workflow
    6. Generate and save the analysis report
    7. Error handling and logging
    """

    # Initialize the execution logging system
    execution_logger = initialize_execution_logger()
    logger.info(
        f"{SUCCESS_ICON} Execution logging system initialized, log directory: {execution_logger.execution_dir}")

    try:
        # ============================================================================
        # 1. Define the LangGraph workflow
        # ============================================================================

        # Create the workflow graph, using AgentState as the state type
        workflow = StateGraph(AgentState)

        # Add the start node - serving as a clear starting point for the parallel branches
        workflow.add_node("start_node", lambda state: state)

        # Add the five core agent nodes
        workflow.add_node("fundamental_analyst", fundamental_agent)  # Fundamental analysis agent
        workflow.add_node("technical_analyst", technical_agent)      # Technical analysis agent
        workflow.add_node("value_analyst", value_agent)             # Valuation analysis agent
        workflow.add_node("news_analyst", news_agent)               # News analysis agent
        workflow.add_node("summarizer", summary_agent)              # Summary agent

        # Set the workflow entry point
        workflow.set_entry_point("start_node")

        # Add the parallel-execution edges - four analysis agents execute in parallel
        workflow.add_edge("start_node", "fundamental_analyst")
        workflow.add_edge("start_node", "technical_analyst")
        workflow.add_edge("start_node", "value_analyst")
        workflow.add_edge("start_node", "news_analyst")

        # Add the convergence edges - all analysis results converge to the summary agent
        # LangGraph ensures that "summarizer" waits for all direct predecessor nodes to complete
        workflow.add_edge("fundamental_analyst", "summarizer")
        workflow.add_edge("technical_analyst", "summarizer")
        workflow.add_edge("value_analyst", "summarizer")
        workflow.add_edge("news_analyst", "summarizer")

        # Add the end edge - the workflow ends after the summary agent completes
        workflow.add_edge("summarizer", END)

        # Compile the workflow
        app = workflow.compile()

        # ============================================================================
        # 2. Implement the command-line interface
        # ============================================================================

        # Create the command-line argument parser
        parser = argparse.ArgumentParser(description="Financial Agent CLI")
        parser.add_argument(
            "--command",
            type=str,
            required=False,  # Changed to optional, to support interactive input
            help="The user query for financial analysis (e.g., '分析 Apple (AAPL)')"
        )
        args = parser.parse_args()

        # Handle the user query input
        if args.command:
            # If the query is provided via a command-line argument
            user_query = args.command
        else:
            # Display the ASCII art splash image and interactive interface
            print("\n")
            print(
                "╔══════════════════════════════════════════════════════════════════════════════╗")
            print(
                "║                                                                              ║")
            print(
                "║      ███████╗██╗███╗   ██╗ █████╗ ███╗   ██╗ ██████╗██╗ █████╗ ██╗          ║")
            print(
                "║      ██╔════╝██║████╗  ██║██╔══██╗████╗  ██║██╔════╝██║██╔══██╗██║          ║")
            print(
                "║      █████╗  ██║██╔██╗ ██║███████║██╔██╗ ██║██║     ██║███████║██║          ║")
            print(
                "║      ██╔══╝  ██║██║╚██╗██║██╔══██║██║╚██╗██║██║     ██║██╔══██║██║          ║")
            print(
                "║      ██║     ██║██║ ╚████║██║  ██║██║ ╚████║╚██████╗██║██║  ██║███████╗      ║")
            print(
                "║      ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚═╝╚═╝  ╚═╝╚══════╝      ║")
            print(
                "║                                                                              ║")
            print(
                "║                █████╗  ██████╗ ███████╗███╗   ██╗████████╗                  ║")
            print(
                "║               ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝                  ║")
            print(
                "║               ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║                     ║")
            print(
                "║               ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║                     ║")
            print(
                "║               ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║                     ║")
            print(
                "║               ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                     ║")
            print(
                "║                                                                              ║")
            print("║                     🏦 US Equity Analysis Agent System                      ║")
            print(
                "║                     Financial Analysis AI Agent System                      ║")
            print(
                "║                                                                              ║")
            print(
                "║    ┌─────────────────────────────────────────────────────────────────┐     ║")
            print("║    │📊 Fundamentals  📈 Technicals  💰 Valuation  📰 News  🤖 Summary│     ║")
            print(
                "║    └─────────────────────────────────────────────────────────────────┘     ║")
            print(
                "║                                                                              ║")
            print(
                "╚══════════════════════════════════════════════════════════════════════════════╝")
            print("\n🔹 This system can perform a comprehensive analysis of US-listed companies, including:")
            print("  • Fundamental analysis - financial condition, profitability, and industry position")
            print("  • Technical analysis - price trends, volume, and technical indicators")
            print("  • Valuation analysis - valuation levels such as P/E and P/B ratios")
            print("  • News analysis - news sentiment analysis and risk assessment")
            print("\n🔹 Multiple natural-language query styles are supported:")
            print("  • Analyze Apple (AAPL)")
            print("  • Take a look at how Tesla stock is doing for me")
            print("  • I'd like to understand the investment value of Microsoft (MSFT)")
            print("  • Is AAPL worth buying?")
            print("  • Analyze the financial condition of NVIDIA (NVDA) for me")
            print("\n🔹 You can describe your analysis needs in any natural language")
            print("🔹 The system will automatically identify the stock name and ticker, and perform a comprehensive analysis")
            print("\n💡 Tip: it is recommended to use a US ticker (e.g. AAPL, MSFT, NVDA) for more accurate analysis results")
            print("\n" + "─" * 78 + "\n")

            # Get the user input
            user_query = input("💬 Please enter your analysis request: ")

            # Ensure the input is not empty
            while not user_query.strip():
                print(f"{ERROR_ICON} Input cannot be empty, please try again!")
                user_query = input("Please enter your analysis request: ")

        # Log the user query to the execution log
        execution_logger.log_agent_start("main", {"user_query": user_query})

        # ============================================================================
        # 3. Natural language processing and stock information extraction
        # ============================================================================

        # Extract the US ticker and company name from the query
        company_name, stock_code = extract_us_stock_info(user_query)

        # Log the extraction result
        logger.info(f"Extracted from query - company name: {company_name}, stock ticker: {stock_code}")

        # ============================================================================
        # 4. Time information processing
        # ============================================================================

        # Get the current time information
        current_datetime = datetime.now()
        current_date_cn = current_datetime.strftime("%Y年%m月%d日")
        current_date_en = current_datetime.strftime("%Y-%m-%d")
        current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                              "星期五", "星期六", "星期日"][current_datetime.weekday()]
        current_time = current_datetime.strftime("%H:%M:%S")

        # Format the complete time information
        current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

        logger.info(f"Current time: {current_time_info}")

        # ============================================================================
        # 5. Prepare the initial state data
        # ============================================================================

        # Prepare the initial state
        initial_data = {
            "query": user_query,
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        }
        
        # Add the company name (if extracted)
        if company_name:
            initial_data["company_name"] = company_name

        # Add the stock ticker (if extracted), and normalize it to a US ticker
        if stock_code:
            normalized_stock_code = normalize_stock_code(stock_code)
            if normalized_stock_code:
                initial_data["stock_code"] = normalized_stock_code

        # Create the initial state for the LangGraph workflow
        initial_state = AgentState(
            messages=[],  # Langchain convention: the message list
            data=initial_data,  # Application-specific data, including the extracted information
            metadata={}  # Other runtime-specific information
        )

        # ============================================================================
        # 6. Execute the workflow
        # ============================================================================

        # Display the analysis start information
        print(f"\n{WAIT_ICON} Starting financial analysis for '{user_query}'...")
        if company_name:
            print(f"{WAIT_ICON} Analyzing company: {company_name}")
        if stock_code:
            print(f"{WAIT_ICON} Stock ticker: {stock_code}")
        logger.info(
            f"Starting financial analysis workflow for query: '{user_query}'")

        # Display the analysis stage prompts
        print(f"\n{WAIT_ICON} Running fundamental analysis...")
        print(f"{WAIT_ICON} Running technical analysis...")
        print(f"{WAIT_ICON} Running valuation analysis...")
        print(f"{WAIT_ICON} Running news analysis...")
        print(f"{WAIT_ICON} This may take a few minutes, please be patient...\n")

        # Invoke the workflow - this is a blocking call that waits for all agents to complete
        final_state = await app.ainvoke(initial_state)
        print(f"{SUCCESS_ICON} Analysis complete!")
        logger.info("Workflow execution completed successfully")

        # ============================================================================
        # 7. Result processing and report generation
        # ============================================================================

        # Extract and print the final report
        if final_state and final_state.get("data") and "final_report" in final_state["data"]:
            print("\n--- Final Analysis Report ---\n")
            # print(final_state["data"]["final_report"])

            # Display the report file path (if available)
            if "report_path" in final_state["data"]:
                print(
                    f"\n{SUCCESS_ICON} Report saved to: {final_state['data']['report_path']}")
                logger.info(
                    f"Report saved to: {final_state['data']['report_path']}")

                # Log the final report to the execution log
                execution_logger.log_final_report(
                    final_state["data"]["final_report"],
                    final_state["data"]["report_path"]
                )
        else:
            print(f"\n{ERROR_ICON} Error: could not retrieve the final report from the workflow.")
            logger.error(
                "Could not retrieve the final report from the workflow")
            print("Debug info - final state content:", final_state)

        # Finalize the execution logging
        finalize_execution_logger(success=True)
        print(f"{SUCCESS_ICON} Execution log saved to: {execution_logger.execution_dir}")

    except Exception as e:
        # ============================================================================
        # 8. Error handling
        # ============================================================================

        print(f"\n{ERROR_ICON} An error occurred during workflow execution: {e}")
        logger.error(f"Error during workflow execution: {e}", exc_info=True)

        # Log the error and finalize the execution logging
        finalize_execution_logger(success=False, error=str(e))
        print(f"{ERROR_ICON} Error log saved to: {get_execution_logger().execution_dir}")


# ============================================================================
# Program entry point
# ============================================================================

if __name__ == "__main__":
    # Run the main function using asyncio
    asyncio.run(main())
