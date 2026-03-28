from __future__ import annotations
import torch
import runpy
import os
import json
import numpy as np
from langchain.tools import tool
from pydantic import BaseModel, Field

from dataflow_agent.state import DFState, MainState
from dataflow_agent.workflow.registry import register
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.toolkits.tool_manager import get_tool_manager
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.optool.op_tools import (
    local_tool_for_get_match_operator_code,
)
from dataflow_agent.agentroles.data_agents.operator_qa_agent import OperatorRAGService

def register_all_tools(builder: GenericGraphBuilder, state: DFState, rag_service: OperatorRAGService):
    
    # ---------------- 前置工具：operator_qa ----------------
    @builder.pre_tool("user_query", "operator_qa")
    def get_user_query(state: MainState) -> str:
        """获取用户查询"""
        return state.request.target or ""
    
    # ---------------- 后置工具：operator_qa ----------------
    class SearchOperatorsInput(BaseModel):
        """搜索算子的输入参数"""
        query: str = Field(description="搜索查询，描述需要的算子功能，如 '过滤文本' '数据清洗'")
        top_k: int = Field(default=5, description="返回结果数量，默认5个")
    
    @builder.post_tool("operator_qa")
    @tool(args_schema=SearchOperatorsInput)
    def search_operators(query: str, top_k: int = 5) -> str:
        result = rag_service.search_and_get_info(query, top_k=top_k)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    class GetOperatorInfoInput(BaseModel):
        operator_name: str = Field(description="要获取信息的算子名称，如 'PromptedFilter'")

    @builder.post_tool("operator_qa")
    @tool(args_schema=GetOperatorInfoInput)
    def get_operator_info(operator_name: str) -> str:
        """获取指定算子的详细信息"""  
        info = rag_service.get_operator_info([operator_name])
        return info

    class GetOperatorSourceInput(BaseModel):
        operator_name: str = Field(description="要获取源码的算子名称，如 'PromptedFilter'")

    @builder.post_tool("operator_qa")
    @tool(args_schema=GetOperatorSourceInput)
    def get_operator_source_code(operator_name: str):
        """获取指定算子的源码内容"""  
        return rag_service.get_operator_source(operator_name)
    
    class GetOperatorParamsInput(BaseModel):
        operator_name: str = Field(description="要获取参数信息的算子名称")
    
    @builder.post_tool("operator_qa")
    @tool(args_schema=GetOperatorParamsInput)
    def get_operator_parameters(operator_name: str):
        """获取指定算子的参数"""
        params = rag_service.get_operator_params(operator_name)
        return json.dumps(params, ensure_ascii=False, indent=2)
    
    # ---------------- 前置工具：write_filter_pipeline ----------------
    @builder.pre_tool("example", "write_filter_pipeline")
    def pre_example_from_matched(state: DFState):
        """
        为写算子提供更强的 in-context 示例：
        将匹配到的所有算子源码（含 模块路径 + import + 类定义）拼接为示例，让 LLM 模仿项目风格。
        优先从 DFState.matched_ops 读取；若为空则回退读取 agent_results。
        """
        names: list[str] = []
        try:
            if isinstance(state.matched_ops, list) and state.matched_ops:
                names = list(dict.fromkeys(state.matched_ops))
            else:
                res = state.agent_results.get("operator_qa", {}).get("results", {})
                names = list(res.get("related_operators", []))
        except Exception:
            names = []

        if not names:
            return ""

        blocks = []
        chunk = 3  
        for i in range(0, len(names), chunk):
            part = names[i:i+chunk]
            try:
                blocks.append(local_tool_for_get_match_operator_code({"match_operators": part}))
            except Exception:
                continue
        code_examples = "\n\n".join([b for b in blocks if b])
        return code_examples

    @builder.pre_tool("target", "write_filter_pipeline")
    def pre_target(state: DFState):
        return state.request.writer_target
  
    # ---------------- 前置工具：code_debugger ----------------
    @builder.pre_tool("pipeline_code", "filter_code_debugger")
    def dbg_get_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("error_trace", "filter_code_debugger")
    def dbg_get_err(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("traceback", "")

    @builder.pre_tool("exec_error", "filter_code_debugger")
    def dbg_get_exec_error(state: DFState):
        return state.execution_result.get("exec_error", "")
    
    @builder.pre_tool("stack_trace", "filter_code_debugger")
    def dbg_get_stack_trace(state: DFState):
        return state.execution_result.get("stack_trace", "")
    
    # ---------------- 前置工具：rewriter ----------------
    @builder.pre_tool("pipeline_code", "filter_rewriter")
    def rw_get_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("stack_trace", "filter_rewriter")
    def rw_get_stack_trace(state: DFState):
        return state.execution_result.get("stack_trace", "")
    
    @builder.pre_tool("debug_reason", "filter_rewriter")
    def rw_get_reason(state: DFState):
        res = state.code_debug_result.get("reason", "")
        return res

    # 为 rewriter 注入数据上下文，辅助其在重写阶段完善自动选键逻辑
    @builder.pre_tool("data_sample", "filter_rewriter")
    def rw_get_data_sample(state: DFState):
        try:
            # 使用有效数据路径，避免取不到样例
            from types import SimpleNamespace as _SN
            default_test_file = f"{PROJDIR}/tests/test.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("samples", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("target", "filter_rewriter")
    def rw_get_target(state: DFState):
        return getattr(state.request, "writer_target", "")

    # ---------------- 前置工具：llm_instantiate ----------------
    @builder.pre_tool("pipeline_code", "filter_llm_instantiate")
    def pre_inst_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("target", "filter_llm_instantiate")
    def pre_inst_target(state: DFState):
        return getattr(state.request, "writer_target", "")

    @builder.pre_tool("dataset_name", "filter_llm_instantiate")
    def pre_inst_dataset_name(state: DFState):
        return getattr(state.request, "dataset_name", "")

    @builder.pre_tool("output_root", "filter_llm_instantiate")
    def pre_inst_output_root(state: DFState):
        return getattr(state.request, "output_root", "")

    @builder.pre_tool("example_data", "filter_llm_instantiate")
    def pre_inst_example(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{PROJDIR}/tests/test.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("samples", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("available_keys", "filter_llm_instantiate")
    def pre_inst_keys(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{PROJDIR}/tests/test.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("available_keys", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("preselected_input_key", "filter_llm_instantiate")
    def pre_inst_preselected_key(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{PROJDIR}/tests/test.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            samples = stats.get("samples", []) if isinstance(stats, dict) else []
            keys = stats.get("available_keys", []) if isinstance(stats, dict) else []
            if not samples or not keys:
                return ""
            # 计算各列的平均字符串长度（基于前2条样例）
            import numpy as _np
            best_k, best_len = "", -1.0
            for k in keys:
                try:
                    vals = [str(s.get(k, "")) for s in samples]
                    avg_len = _np.mean([len(v) for v in vals]) if vals else 0.0
                except Exception:
                    avg_len = 0.0
                if avg_len > best_len:
                    best_k, best_len = k, avg_len
            return best_k
        except Exception:
            return ""

    @builder.pre_tool("test_data_path", "filter_llm_instantiate")
    def pre_inst_test_path(state: DFState):
        try:
            default_test_file = f"{PROJDIR}/tests/test.jsonl"
            return getattr(state.request, 'json_file', '') or default_test_file
        except Exception:
            return ""

    @builder.pre_tool("reference_operator", "filter_llm_instantiate")
    def pre_example_from_matched(state: DFState):
        names: list[str] = []
        try:
            if isinstance(state.matched_ops, list) and state.matched_ops:
                names = list(dict.fromkeys(state.matched_ops))
            else:
                res = state.agent_results.get("operator_qa", {}).get("results", {})
                names = list(res.get("related_operators", []))
        except Exception:
            names = []

        if not names:
            return ""

        blocks = []
        chunk = 3  
        for i in range(0, len(names), chunk):
            part = names[i:i+chunk]
            try:
                blocks.append(local_tool_for_get_match_operator_code({"match_operators": part}))
            except Exception:
                continue
        code_examples = "\n\n".join([b for b in blocks if b])
        return code_examples
    

    