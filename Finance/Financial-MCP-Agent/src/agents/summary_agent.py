"""
Summary Agent: Consolidates analyses from other agents into a final report.
"""
import os
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI  # Restore the OpenAI import
import re

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(override=True)

logger = setup_logger(__name__)


def truncate_report_at_baseline_time(report_content: str, current_time_info: str) -> str:
    """
    Use a regular expression to truncate the report, stopping after the "分析基准时间" line

    Args:
        report_content: the full report content
        current_time_info: the current time information

    Returns:
        The truncated report content
    """
    # Build several possible "分析基准时间" patterns
    baseline_patterns = [
        rf'分析基准时间[：:]\s*{re.escape(current_time_info)}',
        rf'分析基准时间[：:]\s*{re.escape(current_time_info)}\s*$',
        rf'基准时间[：:]\s*{re.escape(current_time_info)}',
        rf'时间基准[：:]\s*{re.escape(current_time_info)}',
        rf'分析时间[：:]\s*{re.escape(current_time_info)}',
        rf'报告时间[：:]\s*{re.escape(current_time_info)}',
        rf'生成时间[：:]\s*{re.escape(current_time_info)}',
        rf'更新时间[：:]\s*{re.escape(current_time_info)}',
        rf'数据时间[：:]\s*{re.escape(current_time_info)}',
        rf'分析基准[：:]\s*{re.escape(current_time_info)}'
    ]
    
    # Try to match the various patterns
    for pattern in baseline_patterns:
        match = re.search(pattern, report_content, re.MULTILINE | re.IGNORECASE)
        if match:
            # Found a match position; truncate to the end of that line
            end_pos = match.end()

            # Find the end position of that line (newline character)
            line_end = report_content.find('\n', end_pos)
            if line_end == -1:
                # If there is no newline character, it is the last line; truncate directly
                truncated_content = report_content[:end_pos].strip()
            else:
                # Truncate to the end of that line
                truncated_content = report_content[:line_end].strip()

            logger.info(f"Truncated the report after the '分析基准时间' line, truncation position: {end_pos}")
            return truncated_content

    # If no matching pattern was found, try to find a line containing the time information
    time_patterns = [
        rf'.*{re.escape(current_time_info)}.*',
        rf'.*{re.escape(current_time_info.split()[0])}.*',  # Match only the date portion
        rf'.*{re.escape(current_time_info.split()[1])}.*'   # Match only the time portion
    ]

    for pattern in time_patterns:
        match = re.search(pattern, report_content, re.MULTILINE | re.IGNORECASE)
        if match:
            end_pos = match.end()
            line_end = report_content.find('\n', end_pos)
            if line_end == -1:
                truncated_content = report_content[:end_pos].strip()
            else:
                truncated_content = report_content[:line_end].strip()

            logger.info(f"Truncated the report after the time information line, truncation position: {end_pos}")
            return truncated_content

    # If nothing was found, return the original content
    logger.warning("Did not find the '分析基准时间' pattern; returning the original report content")
    return report_content


def load_finr1_model(model_path="/root/code/Finance/FinR1"):
    """Load the FinR1 model"""
    logger.info(f"{WAIT_ICON} Loading FinR1 model from {model_path}...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        model.eval()
        logger.info(f"{SUCCESS_ICON} FinR1 model loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"{ERROR_ICON} Failed to load FinR1 model: {e}")
        raise e


def generate_report_with_finr1(model, tokenizer, prompt, max_new_tokens=5000):
    """Generate a report using the FinR1 model"""

    try:
        import torch

        # Encode the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate the prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the generated report portion (remove the input prompt)
        # Method 1: try to remove the input prompt via string matching
        if prompt in generated_text:
            report = generated_text[len(prompt):].strip()
        else:
            # Method 2: if string matching fails, try to extract by token length
            input_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])
            output_length = len(outputs[0])

            if output_length > input_length:
                # Keep only the newly generated portion
                new_tokens = outputs[0][input_length:]
                report = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            else:
                # If it cannot be determined, return the full text but attempt to clean it
                report = generated_text.strip()

        return report

    except Exception as e:
        logger.error(f"{ERROR_ICON} Error generating report with FinR1: {e}")
        raise e


def get_model_choice():
    """Get the model choice, defaulting to API"""
    # The model choice can be controlled via an environment variable
    model_choice = os.getenv("USE_LOCAL_MODEL", "api").lower()
    return model_choice


async def summary_agent(state: AgentState) -> Dict[str, Any]:
    """
    Consolidate the results of the fundamental, technical, and valuation analyses
    Use the LLM to generate the final comprehensive report
    """
    logger.info(f"{WAIT_ICON} SummaryAgent: Starting to consolidate analyses.")

    # Get the execution logger, used to record the Agent's execution process
    execution_logger = get_execution_logger()
    agent_name = "summary_agent"

    # Extract the current data, messages, and user query from the state
    current_data = state.get("data", {})
    messages = state.get("messages", [])
    user_query = current_data.get("query", "")

    # Record the start of the Agent's execution, including the available analysis types
    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "available_analyses": {
            "fundamental": "fundamental_analysis" in current_data,
            "technical": "technical_analysis" in current_data,
            "value": "value_analysis" in current_data,
            "news": "news_analysis" in current_data
        },
        "input_data_keys": list(current_data.keys())
    })

    # Record the Agent start time, used to compute the execution duration
    agent_start_time = time.time()

    # Get the analysis results from the previous Agents
    fundamental_analysis = current_data.get(
        "fundamental_analysis", "Not available")
    technical_analysis = current_data.get(
        "technical_analysis", "Not available")
    value_analysis = current_data.get("value_analysis", "Not available")
    news_analysis = current_data.get("news_analysis", "Not available")

    # Handle the error information from each analysis
    errors = []
    if "fundamental_analysis_error" in current_data:
        errors.append(
            f"Fundamental Analysis Error: {current_data['fundamental_analysis_error']}")
    if "technical_analysis_error" in current_data:
        errors.append(
            f"Technical Analysis Error: {current_data['technical_analysis_error']}")
    if "value_analysis_error" in current_data:
        errors.append(
            f"Value Analysis Error: {current_data['value_analysis_error']}")
    if "news_analysis_error" in current_data:
        errors.append(
            f"News Analysis Error: {current_data['news_analysis_error']}")

    # Basic stock identification information
    stock_code = current_data.get("stock_code", "Unknown Stock")
    company_name = current_data.get("company_name", "Unknown Company")

    try:
        # Get the model choice
        model_choice = get_model_choice()
        logger.info(f"{WAIT_ICON} SummaryAgent: Using model choice: {model_choice}")

        # Get the current time information, used for the time annotations in the report
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")

        # Prepare the system prompt for the summary
        system_prompt = f"""
        You are a professional financial analyst, responsible for creating comprehensive, in-depth stock analysis reports.

        **Important time information: the current actual time is {current_time_info}**
        **Analysis baseline date: {current_date}**

        This is the real current time, not your training data cutoff. When generating the report, please:
        - Judge the timeliness of the data based on the actual current time
        - Correctly label time concepts such as "最新", "近期", "历史"
        - Clearly mark the analysis time baseline in the report as: {current_date}
        - Base all time-related descriptions on this actual date

        Your task is to synthesize four different analysis results:
        1. 基本面分析 - focused on the financial statements, business model, and company fundamentals
        2. 技术分析 - focused on price trends, volume patterns, and technical indicators
        3. 估值分析 - focused on valuation metrics and relative value
        4. 新闻分析 - focused on market sentiment, significant events, and the impact of media coverage on the share price

        Please create a clearly structured, coherent report that integrates the insights from all four analyses.
        Even if some analysis data is incomplete or missing, please provide the best possible comprehensive analysis based on the available information.

        **Strictly follow the report format and structure below:**
        
        # [公司名称]([股票代码]) 综合分析报告
        
        ## 执行摘要
        [提供简明扼要的总体分析和投资建议，包括风险等级和预期回报]
        
        ## 公司概况
        [简要介绍公司的业务、行业地位、主要产品或服务]
        
        ## 基本面分析
        [详细分析公司财务状况、盈利能力、成长性、资产负债情况等]
        
        ## 技术分析
        [详细分析价格趋势、技术指标、支撑位和阻力位、交易量等]
        
        ## 估值分析
        [详细分析估值指标、与行业平均水平比较、历史估值水平、股息收益率等]
        
        ## 新闻分析
        [详细分析市场情绪、重要新闻事件、媒体报道、分析师评级变化等对股价的影响]
        
        ## 综合评估
        [分析不同分析方法之间的一致点和分歧点，提供更全面的投资视角]
        
        ## 风险因素
        [详细分析潜在的风险因素，包括市场风险、行业风险、公司特定风险等]
        
        ## 投资建议
        [提供明确的投资建议，包括目标价格、投资时间范围、适合的投资者类型等]
        
        ## 附录：数据来源与限制
        [说明数据来源，以及分析过程中遇到的任何数据限制或缺失]

        The output must be in valid Markdown format, using appropriate headings, bullet points, and formatting.
        Do not include any code block markers such as ```markdown or ```; output pure Markdown content directly.

        Use professional financial language, but keep it readable. The report should be comprehensive and in-depth, with enough detail and data support,
        while focusing on the most important insights to help investors make decisions.

        **Important reminders:**
        - Please clearly mark the analysis baseline time at the end of the report: {current_time_info}
        - Judge the timeliness of all data based on this actual time
        - Avoid using vague time concepts; make judgments based on the actual current time
        - Strictly follow the format and structure above, ensuring every section has substantive content

        If some analysis data is incomplete or contains errors, please clearly state this in the report, and provide valuable analysis based on the available information as much as possible.

        请用简体中文输出。
        """

        # Prepare the user prompt for the summary
        user_prompt = f"""
        Please create a comprehensive analysis report for {company_name} ({stock_code}) based on the following analyses.
        
        Original user query: {user_query}
        
        FUNDAMENTAL ANALYSIS:
        {fundamental_analysis}
        
        TECHNICAL ANALYSIS:
        {technical_analysis}
        
        VALUE ANALYSIS:
        {value_analysis}
        
        NEWS ANALYSIS:
        {news_analysis}
        
        {"ANALYSIS ISSUES:" if errors else ""}
        {". ".join(errors) if errors else ""}
        
        IMPORTANT: Your output MUST be in valid Markdown format with proper headings, bullet points, 
        and formatting. Include a clear recommendation section at the end.
        
        DO NOT include any code block markers like ```markdown or ``` in your output.
        Just write pure Markdown content directly.
        """

        # Decide which approach to use for generating the report based on the model choice
        if model_choice == "local":
            # Use the local FinR1 model
            logger.info(f"{WAIT_ICON} SummaryAgent: Using local FinR1 model...")

            # Record the model configuration information
            model_config = {
                "model": "FinR1",
                "temperature": 0.5,
                "max_tokens": 5000,
                "model_path": "/root/code/Finance/FinR1"
            }

            # Load the FinR1 model
            model, tokenizer = load_finr1_model()

            # Combine the full prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Record the LLM interaction start time
            llm_start_time = time.time()

            # Use the FinR1 model to generate the final report
            final_report = generate_report_with_finr1(model, tokenizer, full_prompt)

            # Record the LLM interaction execution time
            llm_execution_time = time.time() - llm_start_time

        else:
            # Use the API interface by default
            logger.info(f"{WAIT_ICON} SummaryAgent: Using OpenAI API...")

            # Create the OpenAI model (using a direct API call, rather than the ReAct framework, for the summary)
            api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
            base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

            # Validate that the required environment variables exist
            if not all([api_key, base_url, model_name]):
                logger.error(
                    f"{ERROR_ICON} SummaryAgent: Missing OpenAI environment variables.")
                current_data["summary_error"] = "Missing OpenAI environment variables."

                # Record the Agent execution failure
                execution_logger.log_agent_complete(agent_name, current_data, time.time(
                ) - agent_start_time, False, "Missing OpenAI environment variables")

                return {"data": current_data, "messages": messages}

            # Record the model configuration information
            model_config = {
                "model": model_name,
                "temperature": 0.5,
                "max_tokens": 5000,
                "api_base": base_url
            }

            # Prepare the summary prompt message list
            summary_prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Use the ChatOpenAI model
            logger.info(f"{WAIT_ICON} SummaryAgent: Creating ChatOpenAI with model {model_name}")
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.5,  # Raise the temperature for more creativity and more natural expression
                max_tokens=5000   # Increase the output length to generate a more detailed comprehensive report
            )

            # Record the LLM interaction start time
            llm_start_time = time.time()

            # Call the LLM to generate the final report
            llm_message = await llm.ainvoke(summary_prompt_messages)
            final_report = llm_message.content

            # Record the LLM interaction execution time
            llm_execution_time = time.time() - llm_start_time

        # Record the LLM interaction details, for subsequent analysis and optimization
        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="summary_generation",
            input_messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            output_content=final_report,
            model_config=model_config,
            execution_time=llm_execution_time
        )

        # Remove any markdown code block markers that may appear
        final_report = final_report.replace(
            "```markdown", "").replace("```", "").strip()

        # Use a regular expression to truncate the content after the "分析基准时间" line
        final_report = truncate_report_at_baseline_time(final_report, current_time_info)

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Final report generated for {company_name} ({stock_code}).")
        logger.debug(f"Final report preview: {final_report[:300]}...")

        # Save the report to a Markdown file
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Process the company name and stock ticker to ensure the file name is meaningful
        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            # Extract a more meaningful name from the user query
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"report_{query_based_name}"
        else:
            # In the normal case, use the company name and stock ticker
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            # Clean the stock ticker to ensure it can be used in a file name
            clean_stock_code = stock_code.replace(".", "-").replace("/", "-")
            safe_file_prefix = f"report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"

        # Ensure the reports directory exists
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)

        # Write the report to a file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Report saved to {report_path}")

        # Return the updated state, including the final report
        current_data["final_report"] = final_report
        current_data["report_path"] = report_path

        # Record the Agent execution success
        total_execution_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {
            "final_report_length": len(final_report),
            "report_path": report_path,
            "report_preview": final_report,
            "llm_execution_time": llm_execution_time,
            "total_execution_time": total_execution_time
        }, total_execution_time, True)

        return {"data": current_data, "messages": messages}

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} SummaryAgent: Error generating final report: {e}", exc_info=True)
        current_data["summary_error"] = f"Error generating final report: {e}"

        # Create a minimal report even when an error occurs
        error_report = f"""
        # Analysis Report for {company_name} ({stock_code})
        
        **Error encountered during report generation**: {e}
        
        ## Available Analysis Fragments:
        
        - Fundamental Analysis: {"Available" if fundamental_analysis != "Not available" else "Not available"}
        - Technical Analysis: {"Available" if technical_analysis != "Not available" else "Not available"}
        - Value Analysis: {"Available" if value_analysis != "Not available" else "Not available"}
        - News Analysis: {"Available" if news_analysis != "Not available" else "Not available"}
        
        Please review the individual analyses directly for more information.
        """
        current_data["final_report"] = error_report

        # Also save the error report to a file
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Process the company name and stock ticker to ensure the file name is meaningful
        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            # Extract a more meaningful name from the user query
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"error_report_{query_based_name}"
        else:
            # In the normal case, use the company name and stock ticker
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            # Clean the stock ticker to ensure it can be used in a file name
            clean_stock_code = stock_code.replace(".", "-").replace("/", "-")
            safe_file_prefix = f"error_report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"

        # Ensure the reports directory exists
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)

        # Write the error report to a file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(error_report)

        logger.info(
            f"{ERROR_ICON} SummaryAgent: Error report saved to {report_path}")
        current_data["report_path"] = report_path

        # Record the Agent execution failure
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - agent_start_time, False, str(e))

        return {"data": current_data, "messages": messages}


# Local test function
async def test_summary_agent():
    """Test function for the summary Agent"""
    from src.utils.state_definition import AgentState

    # Example state used for testing, containing simulated analysis results
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析 Apple (AAPL)",
            "stock_code": "AAPL",
            "company_name": "Apple",
            "fundamental_analysis": "Apple基本面分析：公司主营业务为消费电子、软件服务和生态系统服务。财务状况稳健，现金流充裕，毛利率和股东回报能力处于大型科技公司前列。",
            "technical_analysis": "Apple技术分析：短期股价围绕主要均线震荡，成交量和动量指标需要结合最新K线进一步确认。",
            "value_analysis": "Apple估值分析：当前估值需结合市盈率、市销率、自由现金流收益率和大型科技公司可比估值综合判断。",
            "news_analysis": "Apple新闻分析：近期新闻需关注新品周期、服务收入、监管政策和供应链变化对市场预期的影响。"
        },
        metadata={}
    )

    # Run the Agent and output the result
    result = await summary_agent(test_state)
    print("Summary Report:")
    print(result.get("data", {}).get("final_report", "No report generated"))
    print(
        f"Report saved to: {result.get('data', {}).get('report_path', 'Not saved')}")

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_summary_agent())
