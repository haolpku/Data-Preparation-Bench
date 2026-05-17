from __future__ import annotations

from typing import Any, Dict, Optional, List

from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.state import DFState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from .filter_base_agent import FilterBaseAgent

log = get_logger(__name__)


class TargetOptimizer(FilterBaseAgent):
    """目标优化器：自动理解并优化用户目标，使其更适合算子检索"""
    
    @property
    def role_name(self) -> str:
        return "target_optimizer"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_target_optimizer"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_target_optimizer"

    # -------------------- Prompt 参数 -------------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        将前置工具结果映射到 prompt 中的占位符：
            {{ target }} – 原始用户目标
        """
        return {
            "target": pre_tool_results.get("target", ""),
            "example_data": pre_tool_results.get("example_data", ""),
            "dataset_name": pre_tool_results.get("dataset_name", ""),
        }

    # -------------------- 前置工具默认值 -----------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "target": "",
            "example_data": "",
            "dataset_name": "",
        }
    
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        约定 LLM 输出格式：
            optimized_target: str      – 优化后的目标
        """
        state.temp_data["optimized_target"] = result
        
        if result.get("optimized_target"):
            original = pre_tool_results.get("target", "")
            optimized = result["optimized_target"]
            log.info(f"目标优化: {original[:50]}... -> {optimized[:50]}...")
            
        super().update_state_result(state, result, pre_tool_results)


def create_target_optimizer(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> TargetOptimizer:
    return TargetOptimizer(tool_manager=tool_manager, **kwargs)