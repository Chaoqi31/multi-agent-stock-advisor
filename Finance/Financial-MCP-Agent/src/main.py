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

# Workflow graph builder (analysts -> debate -> risk disclosure -> summary)
from src.graph.build_graph import build_workflow
from src.memory.advisory_memory import AdvisoryMemory

# LangGraph checkpointer for resumable runs
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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

        # Build the workflow graph: memory retrieval -> four parallel analysts ->
        # bull/bear debate -> research manager -> risk disclosure -> summarizer.
        # It is compiled later with a checkpointer so runs can be resumed by thread_id.
        workflow = build_workflow()

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
        parser.add_argument(
            "--resume", type=str, default=None, metavar="THREAD_ID",
            help="Resume an interrupted run from its last checkpoint, by thread_id"
        )
        parser.add_argument(
            "--reflect", type=str, default=None, metavar="THREAD_ID",
            help="Backfill the realized outcome of a past run into advisory memory, then exit"
        )
        parser.add_argument(
            "--outcome", type=str, default=None,
            help="Outcome text to store with --reflect (e.g. 'AAPL +6%% over 5 trading days')"
        )
        parser.add_argument(
            "--max-debate-rounds", type=int, default=None,
            help="Number of bull/bear debate rounds (overrides the MAX_DEBATE_ROUNDS env var)"
        )
        args = parser.parse_args()

        # Allow the debate depth to be set from the CLI
        if args.max_debate_rounds is not None:
            os.environ["MAX_DEBATE_ROUNDS"] = str(args.max_debate_rounds)

        # Reflection path: backfill a realized outcome into memory for a past run, then exit.
        if args.reflect:
            outcome = args.outcome or "（未提供具体结果文本）"
            stored = AdvisoryMemory().update_with_outcome(args.reflect, outcome)
            if stored:
                print(f"{SUCCESS_ICON} Reflection stored for run '{args.reflect}'.")
            else:
                print(f"{ERROR_ICON} No stored run found for thread_id '{args.reflect}'.")
            finalize_execution_logger(success=stored)
            return

        # Handle the user query input
        if args.resume:
            # Resuming: no new query; the saved checkpoint carries the prior state.
            user_query = None
            logger.info(f"Resuming previous run with thread_id={args.resume}")
        elif args.command:
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

        # Extract the US ticker and company name from the query (skipped when resuming)
        company_name, stock_code = extract_us_stock_info(user_query) if user_query else (None, None)

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

        # Derive the run's thread_id (resume reuses the supplied one) for checkpointing.
        thread_id = args.resume or (
            f"{initial_data.get('stock_code', 'NA')}_{current_datetime.strftime('%Y%m%d_%H%M%S')}"
        )

        # Display the analysis start information
        if not args.resume:
            print(f"\n{WAIT_ICON} Starting financial analysis for '{user_query}'...")
            if company_name:
                print(f"{WAIT_ICON} Analyzing company: {company_name}")
            if stock_code:
                print(f"{WAIT_ICON} Stock ticker: {stock_code}")
            print(f"\n{WAIT_ICON} Running the four analysts in parallel, then bull/bear debate,")
            print(f"{WAIT_ICON} risk disclosure and the final summary. This may take a few minutes...\n")
        else:
            print(f"\n{WAIT_ICON} Resuming run '{thread_id}' from its last checkpoint...\n")
        print(f"{WAIT_ICON} Run thread_id: {thread_id}  (use --resume / --reflect with this id)")
        logger.info(f"Starting workflow (thread_id={thread_id}, resume={bool(args.resume)})")

        # Compile with a SQLite checkpointer so an interrupted run can be resumed by thread_id.
        # Passing None as the input resumes from the last saved checkpoint.
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoints.sqlite")

        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
            app = workflow.compile(checkpointer=checkpointer)
            run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
            workflow_input = None if args.resume else initial_state
            final_state = await app.ainvoke(workflow_input, config=run_config)

        print(f"{SUCCESS_ICON} Analysis complete!")
        logger.info("Workflow execution completed successfully")

        # Persist this run's situation into cross-run memory for future calibration.
        try:
            final_data = (final_state or {}).get("data", {})
            if final_data.get("final_report"):
                AdvisoryMemory().store(
                    thread_id=thread_id,
                    ticker=final_data.get("stock_code", ""),
                    situation=final_data.get("final_report", "")[:2000],
                    recommendation=final_data.get("debate_conclusion", "")[:1000],
                    date=final_data.get("current_date", ""),
                )
        except Exception as mem_exc:  # noqa: BLE001 - memory is best-effort
            logger.warning(f"Could not store advisory memory: {mem_exc}")

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
