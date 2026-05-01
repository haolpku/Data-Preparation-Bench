from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

from dataflow_agent.state import DFState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger

from .filter_base_agent import FilterBaseAgent
log = get_logger(__name__)

class FilterPipelineWriter(FilterBaseAgent):
    @property
    def role_name(self) -> str:
        return "write_filter_pipeline"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_write_filter_pipeline"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_write_filter_pipeline"
    
    # ---------------- Prompt 参数 --------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        # 维持模板仅使用 {example},{target}；将数据上下文并入 example
        example = pre_tool_results.get("example", "")
        data_sample = pre_tool_results.get("data_sample", [])
        available_keys = pre_tool_results.get("available_keys", [])
        try:
            import json
            preview = (
                "\n\n# 数据样例预览\n"
                + json.dumps(data_sample, ensure_ascii=False)
                + "\n# 数据中可用的字段（keys）\n"
                + json.dumps(available_keys, ensure_ascii=False)
                + "\n# 运行约束\n"
                + (
                    "调试阶段不会传入 input_key。请将pipline运行接口定义为 forward(self)"
                    "而forward内部是多个算子的组合, 需与相应的算子一致;"
                    "operator_x.run(storage.step(): DataFlowStorage, param1: str | None = None, param2: str | None = None, ...);"
                )
            )
            example = (example or "") + preview
        except Exception:
            pass
        return {
            "example": example,
            "target": pre_tool_results.get("target", ""),
            "example_data": pre_tool_results.get("example_data", ""),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "example": "",
            "target": "",
            "example_data": "",
        }
        
    def _dump_code(self, state: DFState, new_code: str) -> Path | None:
        """
        将新代码写入目标文件。如果未提供路径，则仅返回 None（不强制落盘）。
        优先顺序：
          1) state.execution_result["file_path"]
          2) state.temp_data["pipeline_file_path"]
        """
        file_path_str: str | None = None
        if isinstance(state.execution_result, dict):
            file_path_str = state.execution_result.get("file_path")
        file_path_str = file_path_str or state.request.get("pipeline_file_path")

        if not file_path_str:
            # 不强制写入临时文件，避免误写；仅提示
            log.info("未提供目标文件路径，跳过写盘（代码保存在 state.draft_operator_code）")
            return None

        file_path = Path(file_path_str)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_code)
            log.info(f"已将新代码写入 {file_path}")
            return file_path
        except Exception as e:
            log.error(f"写入文件 {file_path} 失败: {e}")
            return None

    # ---------------- 更新 DFState -------------------
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        code_str = ""
        if isinstance(result, dict):
            code_str = result.get("code", "")
        # 将生成代码写入状态，并同步到 temp_data 以便后续执行/调试节点复用
        state.draft_operator_code = code_str
        if code_str:
            saved_path = self._dump_code(state, code_str)
            try:
                state.temp_data["pipeline_code"] = code_str
                if saved_path is not None:
                    state.temp_data["pipeline_file_path"] = str(saved_path)
            except Exception:
                pass
        super().update_state_result(state, result, pre_tool_results)
    
def create_writer(tool_manager: Optional[ToolManager] = None, **kwargs) -> FilterPipelineWriter:
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return FilterPipelineWriter(tool_manager=tool_manager, **kwargs)