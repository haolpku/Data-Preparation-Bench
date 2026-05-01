from __future__ import annotations

from typing import Any, Dict, Optional, List

from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.state import DFState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
log = get_logger(__name__)

from .filter_base_agent import FilterBaseAgent
class FilterCodeDebugger(FilterBaseAgent):
    @property
    def role_name(self) -> str:
        return "filter_code_debugger"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_filter_code_debugging"
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_filter_code_debugging"

    # -------------------- Prompt 参数 -------------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        将前置工具结果映射到 prompt 中的占位符：
            {{ pipeline_code }}   – 需要调试的代码
            {{ error_trace }}     – 本次执行捕获的异常信息
        """
        return {
            "pipeline_code": pre_tool_results.get("pipeline_code", ""),
            "error_trace": pre_tool_results.get("error_trace", ""),
            "exec_error": pre_tool_results.get("exec_error", ""),
            "stack_trace": pre_tool_results.get("stack_trace", ""),
        }

    # -------------------- 前置工具默认值 -----------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "pipeline_code": "",
            "error_trace": "",
            "exec_error": "",
            "stack_trace": "",
        }
    
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        约定 LLM 输出格式：
            reason: str      – 调试分析
        """
        state.code_debug_result = result
        super().update_state_result(state, result, pre_tool_results)
    
def create_code_debugger(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> FilterCodeDebugger:
    return FilterCodeDebugger(tool_manager=tool_manager, **kwargs)