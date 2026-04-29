from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

from dataflow_agent.state import DFState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger

from .filter_base_agent import FilterBaseAgent

log = get_logger(__name__)

class FilterPipelineInstantiater(FilterBaseAgent):
    @property
    def role_name(self) -> str:
        return "filter_llm_instantiate"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_filter_llm_instantiate"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_filter_llm_instantiate"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target": pre_tool_results.get("target", ""),
            "reference_operators": pre_tool_results.get("reference_operator", []),
            "pipeline_code": pre_tool_results.get("pipeline_code", ""),
            "example_data": pre_tool_results.get("example_data", []),
            "available_keys": pre_tool_results.get("available_keys", []),
            "preselected_input_key": pre_tool_results.get("preselected_input_key", ""),
            "test_data_path": pre_tool_results.get("test_data_path", ""),
            "dataset_name": pre_tool_results.get("dataset_name", ""),
            "output_root": pre_tool_results.get("output_root", ""),
            "api_url": pre_tool_results.get("api_url", ""),
            "api_key": pre_tool_results.get("api_key", ""),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "target": "",
            "reference_operator": "",
            "pipeline_code": "",
            "example_data": [],
            "available_keys": [],
            "preselected_input_key": "",
            "test_data_path": "",
            "dataset_name": "",
            "output_root": "",
            "api_url": "",
            "api_key": "",
        }
        
    def _dump_code(self, state: DFState, code: str) -> Optional[Path]:
        file_path = state.temp_data.get("pipeline_file_path") or getattr(state.request, "pipeline_file_path", "")
        if not file_path:
            return None
        p = Path(file_path)
        try:
            p.write_text(code, encoding="utf-8")
            return p
        except Exception:
            return None
        
    def update_state_result(self, state, result, pre_tool_results):
        code = result.get("code", "") if isinstance(result, dict) else ""
#         code = f"""from dataflow.utils.registry import OPERATOR_REGISTRY
# if hasattr(OPERATOR_REGISTRY, "_get_all"):
#     OPERATOR_REGISTRY._get_all()

# {code}
# """
        if code:
            state.temp_data["pipeline_code"] = code
            state.draft_operator_code = code
            saved = self._dump_code(state, code)
            if saved:
                state.temp_data["pipeline_file_path"] = str(saved)

        setattr(state, self.role_name.lower(), result)
        super().update_state_result(state, result, pre_tool_results)
        
def create_llm_instantiator(tool_manager: Optional[ToolManager] = None, **kwargs) -> FilterPipelineInstantiater:
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return FilterPipelineInstantiater(tool_manager=tool_manager, **kwargs)
