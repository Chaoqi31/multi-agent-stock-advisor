"""
Execution logging system - creates a separate log folder for each run.
Records all interactions between agents and the LLM, including inputs, outputs, execution time, etc.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid


class ExecutionLogger:
    """Execution logger."""

    def __init__(self, base_log_dir: str = "logs"):
        """
        Initialize the execution logger.

        Args:
            base_log_dir: the base log directory
        """
        self.base_log_dir = Path(base_log_dir)
        self.execution_id = self._generate_execution_id()
        self.execution_dir = self._create_execution_dir()
        self.start_time = time.time()

        # Record execution start information
        self._log_execution_start()

    def _generate_execution_id(self) -> str:
        """Generate a unique execution ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"

    def _create_execution_dir(self) -> Path:
        """Create the log directory for this execution."""
        execution_dir = self.base_log_dir / self.execution_id
        execution_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (execution_dir / "agents").mkdir(exist_ok=True)
        (execution_dir / "llm_interactions").mkdir(exist_ok=True)
        (execution_dir / "tools").mkdir(exist_ok=True)
        (execution_dir / "reports").mkdir(exist_ok=True)

        return execution_dir

    def _log_execution_start(self):
        """Record execution start information."""
        start_info = {
            "execution_id": self.execution_id,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": self.start_time,
            "environment": {
                "python_version": os.sys.version,
                "working_directory": os.getcwd(),
                "environment_variables": {
                    "OPENAI_COMPATIBLE_MODEL": os.getenv("OPENAI_COMPATIBLE_MODEL", "Not Set"),
                    "OPENAI_COMPATIBLE_BASE_URL": os.getenv("OPENAI_COMPATIBLE_BASE_URL", "Not Set"),
                    "OPENAI_COMPATIBLE_API_KEY": "***" if os.getenv("OPENAI_COMPATIBLE_API_KEY") else "Not Set"
                }
            }
        }

        self._save_json(start_info, "execution_info.json")

    def log_agent_start(self, agent_name: str, input_data: Dict[str, Any]):
        """Record the start of an agent's execution."""
        agent_log = {
            "agent_name": agent_name,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": time.time(),
            "input_data": input_data,
            "status": "started"
        }

        agent_file = f"agents/{agent_name}_execution.json"
        self._save_json(agent_log, agent_file)

        return agent_log

    def log_agent_complete(self, agent_name: str, output_data: Dict[str, Any],
                           execution_time: float, success: bool = True, error: str = None):
        """Record the completion of an agent's execution."""
        # Read the existing agent log
        agent_file = f"agents/{agent_name}_execution.json"
        agent_log = self._load_json(agent_file) or {}

        # Update the completion information
        agent_log.update({
            "end_time": datetime.now().isoformat(),
            "end_timestamp": time.time(),
            "execution_time_seconds": execution_time,
            "output_data": output_data,
            "success": success,
            "error": error,
            "status": "completed" if success else "failed"
        })

        self._save_json(agent_log, agent_file)
        return agent_log

    def log_llm_interaction(self, agent_name: str, interaction_type: str,
                            input_messages: List[Dict], output_content: str,
                            model_config: Dict[str, Any], execution_time: float,
                            token_usage: Optional[Dict] = None):
        """Record the details of an LLM interaction."""
        interaction_id = str(uuid.uuid4())[:8]
        interaction_log = {
            "interaction_id": interaction_id,
            "agent_name": agent_name,
            # "react_agent", "summary", etc.
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "model_config": model_config,
            "input": {
                "messages": input_messages,
                "message_count": len(input_messages),
                "total_input_length": sum(len(str(msg.get("content", ""))) for msg in input_messages)
            },
            "output": {
                "content": output_content,
                "content_length": len(output_content)
            },
            "performance": {
                "execution_time_seconds": execution_time,
                "token_usage": token_usage
            }
        }

        # Save to the LLM interactions directory
        interaction_file = f"llm_interactions/{agent_name}_{interaction_type}_{interaction_id}.json"
        self._save_json(interaction_log, interaction_file)

        # Also save a plain-text version of the input and output for easy viewing
        self._save_text(
            f"=== INPUT MESSAGES ===\n{json.dumps(input_messages, ensure_ascii=False, indent=2)}\n\n"
            f"=== OUTPUT CONTENT ===\n{output_content}",
            f"llm_interactions/{agent_name}_{interaction_type}_{interaction_id}.txt"
        )

        return interaction_log

    def log_tool_usage(self, agent_name: str, tool_name: str, tool_input: Dict,
                       tool_output: Any, execution_time: float, success: bool = True, error: str = None):
        """Record tool usage."""
        tool_log = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "tool_name": tool_name,
            "input": tool_input,
            "output": str(tool_output)[:1000] + "..." if len(str(tool_output)) > 1000 else str(tool_output),
            "execution_time_seconds": execution_time,
            "success": success,
            "error": error
        }

        # Append to the tool usage log file
        tools_file = f"tools/{agent_name}_tools.jsonl"
        self._append_jsonl(tool_log, tools_file)

        return tool_log

    def log_final_report(self, report_content: str, report_path: str):
        """Record the final generated report."""
        report_log = {
            "timestamp": datetime.now().isoformat(),
            "report_path": report_path,
            "report_length": len(report_content),
            "report_preview": report_content
        }

        # Save the report log
        self._save_json(report_log, "reports/final_report_info.json")

        # Save a copy of the report
        self._save_text(report_content, "reports/final_report.md")

        return report_log

    def finalize_execution(self, success: bool = True, error: str = None):
        """Finalize the execution log."""
        end_time = time.time()
        total_execution_time = end_time - self.start_time

        # Read the execution information
        execution_info = self._load_json("execution_info.json") or {}

        # Update the completion information
        execution_info.update({
            "end_time": datetime.now().isoformat(),
            "end_timestamp": end_time,
            "total_execution_time_seconds": total_execution_time,
            "success": success,
            "error": error,
            "status": "completed" if success else "failed"
        })

        # Generate the execution summary
        summary = self._generate_execution_summary()
        execution_info["summary"] = summary

        self._save_json(execution_info, "execution_info.json")

        # Generate a readable summary report
        self._generate_readable_summary(execution_info)

        return execution_info

    def _generate_execution_summary(self) -> Dict[str, Any]:
        """Generate the execution summary."""
        summary = {
            "agents_executed": [],
            "llm_interactions_count": 0,
            "tools_used_count": 0,
            "total_files_created": 0
        }

        # Tally agent executions
        agents_dir = self.execution_dir / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*_execution.json"):
                agent_data = self._load_json(f"agents/{agent_file.name}")
                if agent_data:
                    summary["agents_executed"].append({
                        "name": agent_data.get("agent_name"),
                        "success": agent_data.get("success", False),
                        "execution_time": agent_data.get("execution_time_seconds", 0)
                    })

        # Tally the number of LLM interactions
        llm_dir = self.execution_dir / "llm_interactions"
        if llm_dir.exists():
            summary["llm_interactions_count"] = len(
                list(llm_dir.glob("*.json")))

        # Tally the number of tool uses
        tools_dir = self.execution_dir / "tools"
        if tools_dir.exists():
            for tool_file in tools_dir.glob("*.jsonl"):
                with open(tool_file, 'r', encoding='utf-8') as f:
                    summary["tools_used_count"] += len(f.readlines())

        # Tally the number of files created
        summary["total_files_created"] = len(
            list(self.execution_dir.rglob("*")))

        return summary

    def _generate_readable_summary(self, execution_info: Dict[str, Any]):
        """Generate a readable summary report."""
        summary_text = f"""
# Execution Summary Report

## Basic Information
- Execution ID: {execution_info['execution_id']}
- Start time: {execution_info['start_time']}
- End time: {execution_info.get('end_time', 'N/A')}
- Total execution time: {execution_info.get('total_execution_time_seconds', 0):.2f} seconds
- Execution status: {'Success' if execution_info.get('success', False) else 'Failed'}

## Environment Information
- Model: {execution_info['environment']['environment_variables']['OPENAI_COMPATIBLE_MODEL']}
- API URL: {execution_info['environment']['environment_variables']['OPENAI_COMPATIBLE_BASE_URL']}

## Execution Statistics
- Number of agents executed: {len(execution_info.get('summary', {}).get('agents_executed', []))}
- Number of LLM interactions: {execution_info.get('summary', {}).get('llm_interactions_count', 0)}
- Number of tool uses: {execution_info.get('summary', {}).get('tools_used_count', 0)}
- Number of files created: {execution_info.get('summary', {}).get('total_files_created', 0)}

## Agent Execution Details
"""

        for agent in execution_info.get('summary', {}).get('agents_executed', []):
            status = '✅ Success' if agent.get('success') else '❌ Failed'
            summary_text += f"- {agent.get('name', 'Unknown')}: {status} (elapsed: {agent.get('execution_time', 0):.2f}s)\n"

        if execution_info.get('error'):
            summary_text += f"\n## Error Information\n{execution_info['error']}\n"

        summary_text += f"\n## Log File Location\n{self.execution_dir}\n"

        self._save_text(summary_text, "EXECUTION_SUMMARY.md")

    def _save_json(self, data: Dict[str, Any], filename: str):
        """Save JSON data."""
        file_path = self.execution_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load JSON data."""
        file_path = self.execution_dir / filename
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _append_jsonl(self, data: Dict[str, Any], filename: str):
        """Append JSONL data."""
        file_path = self.execution_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def _save_text(self, content: str, filename: str):
        """Save text content."""
        file_path = self.execution_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


# Global execution logger instance
_execution_logger: Optional[ExecutionLogger] = None


def get_execution_logger() -> ExecutionLogger:
    """Get the global execution logger."""
    global _execution_logger
    if _execution_logger is None:
        _execution_logger = ExecutionLogger()
    return _execution_logger


def initialize_execution_logger(base_log_dir: str = "logs") -> ExecutionLogger:
    """Initialize the execution logger."""
    global _execution_logger
    _execution_logger = ExecutionLogger(base_log_dir)
    return _execution_logger


def finalize_execution_logger(success: bool = True, error: str = None):
    """Finalize the execution log."""
    global _execution_logger
    if _execution_logger:
        _execution_logger.finalize_execution(success, error)
        _execution_logger = None
