"""
Log viewer - for viewing and analyzing execution logs.
Provides multiple ways to view: latest execution, by time range, by execution ID, etc.
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


class LogViewer:
    """Execution log viewer."""

    def __init__(self, base_log_dir: str = "logs"):
        """
        Initialize the log viewer.

        Args:
            base_log_dir: the base log directory
        """
        self.base_log_dir = Path(base_log_dir)

    def list_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List the most recent execution records.

        Args:
            limit: the maximum number of records to return

        Returns:
            a list of execution records
        """
        executions = []

        if not self.base_log_dir.exists():
            return executions

        # Get all execution directories
        execution_dirs = [d for d in self.base_log_dir.iterdir() if d.is_dir()]

        # Sort by creation time
        execution_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        for exec_dir in execution_dirs[:limit]:
            execution_info_file = exec_dir / "execution_info.json"
            if execution_info_file.exists():
                try:
                    with open(execution_info_file, 'r', encoding='utf-8') as f:
                        execution_info = json.load(f)

                    # Add the directory path information
                    execution_info["log_directory"] = str(exec_dir)
                    executions.append(execution_info)
                except Exception as e:
                    print(f"Error reading {execution_info_file}: {e}")

        return executions

    def get_execution_details(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the detailed information for a specific execution.

        Args:
            execution_id: the execution ID

        Returns:
            the detailed execution information
        """
        execution_dir = self.base_log_dir / execution_id
        if not execution_dir.exists():
            return None

        details = {}

        # Read the execution information
        execution_info_file = execution_dir / "execution_info.json"
        if execution_info_file.exists():
            with open(execution_info_file, 'r', encoding='utf-8') as f:
                details["execution_info"] = json.load(f)

        # Read the agent execution information
        agents_dir = execution_dir / "agents"
        if agents_dir.exists():
            details["agents"] = {}
            for agent_file in agents_dir.glob("*_execution.json"):
                agent_name = agent_file.stem.replace("_execution", "")
                with open(agent_file, 'r', encoding='utf-8') as f:
                    details["agents"][agent_name] = json.load(f)

        # Read the LLM interaction information
        llm_dir = execution_dir / "llm_interactions"
        if llm_dir.exists():
            details["llm_interactions"] = []
            for llm_file in llm_dir.glob("*.json"):
                with open(llm_file, 'r', encoding='utf-8') as f:
                    details["llm_interactions"].append(json.load(f))

        # Read the tool usage information
        tools_dir = execution_dir / "tools"
        if tools_dir.exists():
            details["tool_usage"] = []
            for tool_file in tools_dir.glob("*.jsonl"):
                with open(tool_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            details["tool_usage"].append(json.loads(line))

        # Read the report information
        reports_dir = execution_dir / "reports"
        if reports_dir.exists():
            report_info_file = reports_dir / "final_report_info.json"
            if report_info_file.exists():
                with open(report_info_file, 'r', encoding='utf-8') as f:
                    details["report_info"] = json.load(f)

        return details

    def print_execution_summary(self, execution_info: Dict[str, Any]):
        """Print the execution summary."""
        print(f"\n{'='*60}")
        print(f"Execution ID: {execution_info.get('execution_id', 'Unknown')}")
        print(f"Start time: {execution_info.get('start_time', 'Unknown')}")
        print(f"End time: {execution_info.get('end_time', 'Running...')}")

        if 'total_execution_time_seconds' in execution_info:
            print(
                f"Total execution time: {execution_info['total_execution_time_seconds']:.2f} seconds")

        status = execution_info.get('status', 'Unknown')
        status_icon = "✅" if execution_info.get('success', False) else "❌"
        print(f"Execution status: {status_icon} {status}")

        if 'environment' in execution_info:
            env = execution_info['environment']['environment_variables']
            print(f"Model used: {env.get('OPENAI_COMPATIBLE_MODEL', 'Unknown')}")

        if 'summary' in execution_info:
            summary = execution_info['summary']
            print(f"\nExecution statistics:")
            print(f"  - Number of agents: {len(summary.get('agents_executed', []))}")
            print(f"  - Number of LLM interactions: {summary.get('llm_interactions_count', 0)}")
            print(f"  - Number of tool uses: {summary.get('tools_used_count', 0)}")
            print(f"  - Number of files created: {summary.get('total_files_created', 0)}")

        if execution_info.get('error'):
            print(f"\n❌ Error information: {execution_info['error']}")

        print(f"Log directory: {execution_info.get('log_directory', 'Unknown')}")

    def print_agent_details(self, agents_info: Dict[str, Any]):
        """Print the agent execution details."""
        print(f"\n{'='*60}")
        print("AGENT EXECUTION DETAILS")
        print(f"{'='*60}")

        for agent_name, agent_data in agents_info.items():
            status_icon = "✅" if agent_data.get('success', False) else "❌"
            print(f"\n{status_icon} {agent_name.upper()}")
            print(f"  Start time: {agent_data.get('start_time', 'Unknown')}")
            print(f"  End time: {agent_data.get('end_time', 'Unknown')}")

            if 'execution_time_seconds' in agent_data:
                print(f"  Execution time: {agent_data['execution_time_seconds']:.2f} seconds")

            if agent_data.get('error'):
                print(f"  ❌ Error: {agent_data['error']}")

            # Show the output preview
            output_data = agent_data.get('output_data', {})
            for key, value in output_data.items():
                if key.endswith('_preview') or key.endswith('_length'):
                    print(f"  {key}: {value}")

    def print_llm_interactions(self, llm_interactions: List[Dict[str, Any]]):
        """Print the LLM interaction details."""
        print(f"\n{'='*60}")
        print("LLM INTERACTION DETAILS")
        print(f"{'='*60}")

        for interaction in llm_interactions:
            print(
                f"\n🤖 {interaction.get('agent_name', 'Unknown')} - {interaction.get('interaction_type', 'Unknown')}")
            print(f"  Time: {interaction.get('timestamp', 'Unknown')}")
            print(
                f"  Model: {interaction.get('model_config', {}).get('model', 'Unknown')}")
            print(
                f"  Execution time: {interaction.get('performance', {}).get('execution_time_seconds', 0):.2f} seconds")

            input_info = interaction.get('input', {})
            print(f"  Number of input messages: {input_info.get('message_count', 0)}")
            print(f"  Input length: {input_info.get('total_input_length', 0)} characters")

            output_info = interaction.get('output', {})
            print(f"  Output length: {output_info.get('content_length', 0)} characters")

    def print_tool_usage(self, tool_usage: List[Dict[str, Any]]):
        """Print the tool usage details."""
        if not tool_usage:
            return

        print(f"\n{'='*60}")
        print("TOOL USAGE DETAILS")
        print(f"{'='*60}")

        for tool_log in tool_usage:
            status_icon = "✅" if tool_log.get('success', True) else "❌"
            print(
                f"\n{status_icon} {tool_log.get('tool_name', 'Unknown')} (by {tool_log.get('agent_name', 'Unknown')})")
            print(f"  Time: {tool_log.get('timestamp', 'Unknown')}")
            print(f"  Execution time: {tool_log.get('execution_time_seconds', 0):.2f} seconds")

            if tool_log.get('error'):
                print(f"  ❌ Error: {tool_log['error']}")

    def show_execution(self, execution_id: str, show_details: bool = True):
        """Show the complete information for a specific execution."""
        details = self.get_execution_details(execution_id)
        if not details:
            print(f"❌ Execution ID not found: {execution_id}")
            return

        # Show the execution summary
        if 'execution_info' in details:
            self.print_execution_summary(details['execution_info'])

        if not show_details:
            return

        # Show the agent details
        if 'agents' in details:
            self.print_agent_details(details['agents'])

        # Show the LLM interaction details
        if 'llm_interactions' in details:
            self.print_llm_interactions(details['llm_interactions'])

        # Show the tool usage details
        if 'tool_usage' in details:
            self.print_tool_usage(details['tool_usage'])

        # Show the report information
        if 'report_info' in details:
            print(f"\n{'='*60}")
            print("REPORT INFORMATION")
            print(f"{'='*60}")
            report_info = details['report_info']
            print(f"Report path: {report_info.get('report_path', 'Unknown')}")
            print(f"Report length: {report_info.get('report_length', 0)} characters")
            print(f"Generated at: {report_info.get('timestamp', 'Unknown')}")

    def show_recent_executions(self, limit: int = 5):
        """Show the most recent execution records."""
        executions = self.list_executions(limit)

        if not executions:
            print("❌ No execution records found")
            return

        print(f"\n📊 {len(executions)} most recent execution records:")
        print(f"{'='*80}")

        for i, execution in enumerate(executions, 1):
            print(f"\n{i}. {execution.get('execution_id', 'Unknown')}")
            status_icon = "✅" if execution.get('success', False) else "❌"
            print(f"   Status: {status_icon} {execution.get('status', 'Unknown')}")
            print(f"   Time: {execution.get('start_time', 'Unknown')}")

            if 'total_execution_time_seconds' in execution:
                print(
                    f"   Elapsed: {execution['total_execution_time_seconds']:.2f} seconds")

            if 'environment' in execution:
                env = execution['environment']['environment_variables']
                print(
                    f"   Model: {env.get('OPENAI_COMPATIBLE_MODEL', 'Unknown')}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Execution log viewer")
    parser.add_argument("--list", "-l", action="store_true", help="List the most recent execution records")
    parser.add_argument("--show", "-s", type=str, help="Show the details of a specific execution ID")
    parser.add_argument("--limit", type=int, default=5, help="Limit on the number of records to list")
    parser.add_argument(
        "--summary-only", action="store_true", help="Show only the summary, not the details")
    parser.add_argument("--log-dir", type=str, default="logs", help="Path to the log directory")

    args = parser.parse_args()

    viewer = LogViewer(args.log_dir)

    if args.show:
        viewer.show_execution(args.show, not args.summary_only)
    elif args.list:
        viewer.show_recent_executions(args.limit)
    else:
        # By default, show the most recent execution records
        viewer.show_recent_executions(args.limit)


if __name__ == "__main__":
    main()
