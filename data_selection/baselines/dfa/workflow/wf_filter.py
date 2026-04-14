from __future__ import annotations
import torch
import runpy
import os
import json
import traceback
import numpy as np
from langchain.tools import tool
from pydantic import BaseModel, Field

from dataflow_agent.state import DFState, MainState
from dataflow_agent.workflow.registry import register
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.toolkits.tool_manager import get_tool_manager
from dataflow_agent.logger import get_logger
# from dataflow_agent.toolkits.optool.op_tools import (
#     local_tool_for_get_match_operator_code,
# )
from dataflow_agent.agentroles.data_agents.operator_qa_agent import (
    OperatorQAAgent,
    OperatorRAGService,
    create_operator_qa_agent,
)
# from tools.register_all import register_all_tools
from dataflow_agent.utils import get_project_root

from dataflow_agent.toolkits.basetool.file_tools import (
    local_tool_for_sample,
)
from dataflow_agent.agentroles.data_agents.operator_qa_agent import OperatorRAGService

from common_agents.filter_pipeline_writer import create_writer
from common_agents.filter_pipeline_instantiator import create_llm_instantiator
from common_agents.filter_rewriter import create_rewriter

from common_agents.filter_code_debugger import create_code_debugger
from utils import (
    split_dataset, 
    create_simple_parallel_script, merge_jsonl_results,
    extract_conversations_to_json, run_python_file
)

PROJDIR = get_project_root()

log = get_logger(__name__)

def local_tool_for_get_match_operator_code(pre_task_result):
    import time
    import sys
    import inspect
    from dataflow.utils.registry import OPERATOR_REGISTRY

    start_time = time.time()
    if not pre_task_result or not isinstance(pre_task_result, dict):
        return "# ❗ pre_task_result is empty, cannot extract operator names"

    _NAME2CLS = {name: cls for name, cls in OPERATOR_REGISTRY}

    blocks = []
    for op_name in pre_task_result.get("match_operators", []):
        cls = _NAME2CLS.get(op_name)
        if cls is None:
            blocks.append(f"# --- {op_name} is not registered in OPERATOR_REGISTRY ---")
            continue
        try:
            module = sys.modules[cls.__module__]
            module_file = getattr(module, '__file__', None)
            
            cls_src = inspect.getsource(cls)
            module_src = inspect.getsource(sys.modules[cls.__module__])
            import_lines = [
                l for l in module_src.splitlines()
                if l.strip().startswith(("import ", "from "))
            ]
            
            
            import_block = "\n".join(import_lines)
            location_info = []
            location_info.append(f"# Operator: {op_name}")
            location_info.append(f"# Class: {cls.__module__}.{cls.__name__}")
            if module_file:
                location_info.append(f"# File: {module_file}")
            
                # 构建完整的代码块
                src_block = "\n".join([
                    "# " + "=" * 60,
                    f"# Source of {op_name}",
                    "# " + "=" * 60,
                    *location_info,
                    "# " + "-" * 60,
                    import_block,
                    "\n",
                    cls_src
                ])
                blocks.append(src_block)
            # src_block = f"# === Source of {op_name} ===\n{import_block}\n\n{cls_src}"
            # blocks.append(src_block)
        except (OSError, TypeError) as e:
            blocks.append(f"# --- Failed to get the source code of {op_name}: {e} ---")
    
    elapsed = time.time() - start_time
    log.info(f"[local_tool_for_get_match_operator_code] Time used: {elapsed:.4f} seconds")
    return "\n\n".join(blocks)

def get_llm_model_name(state, default="gpt-4o"):
    try:
        req = getattr(state, "request", None)
        for attr in ("model", "model_name", "llm_model"):
            val = getattr(req, attr, None)
            if val: return val.strip()
        return state.temp_data.get("model", default)
    except:
        return default

def _get_llm_model_name(state: DFState, default: str = "gpt-4o") -> str:
    """Resolve model name from request/state to avoid hard-coded defaults."""
    try:
        req = getattr(state, "request", None)
        for attr in ("model", "model_name", "llm_model", "chat_model"):
            val = getattr(req, attr, None) if req is not None else None
            if isinstance(val, str) and val.strip():
                return val.strip()
        tmp = getattr(state, "temp_data", {}) or {}
        val = tmp.get("model") or tmp.get("model_name")
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    return default
    
@register("filter")
def create_filter_graph(state: DFState):
    builder = GenericGraphBuilder(state_model=DFState, entry_point="preprocess_data_node")
    
    # ==========================================
    # 1. Shared Services & RAG Initialization
    # ==========================================
    rag_service = OperatorRAGService(embedding_api_url=state.request.chat_api_url + "/embeddings")
    from dataflow_agent.graphbuilder.message_history import AdvancedMessageHistory
    shared_message_history = AdvancedMessageHistory()
    # ==========================================
    # 2. Tool Bindings (Pre-Tools & Post-Tools)
    # ==========================================
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
    
    @builder.pre_tool("example_data", "write_filter_pipeline")
    def pre_inst_example(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{PROJDIR}/tests/test.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("samples", []) if isinstance(stats, dict) else []
        except Exception:
            return []
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

    @builder.pre_tool("api_url", "filter_llm_instantiate")
    def pre_inst_api_url(state: DFState):
        try:
            return getattr(state.request, 'chat_api_url', '') 
        except Exception:
            return ""
        
    @builder.pre_tool("api_key", "filter_llm_instantiate")
    def pre_inst_api_key(state: DFState):
        try:
            return getattr(state.request, 'api_key', '') 
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
    
    # ==========================================
    # 3. Node Implementations
    # ==========================================
    async def preprocess_data_node(s: DFState) -> DFState:
        from types import SimpleNamespace as _SN
        from dataflow.utils.storage import FileStorage
        
        dataset_path = getattr(s.request, "real_json_file", "")
        dataset_name = dataset_path.split("/")[-1].split(".")[0] if dataset_path else "dataset"
        cache_type = os.path.splitext(dataset_path)[1][1:]

        output_json_path = s.request.real_json_file
        base, ext = os.path.splitext(output_json_path)
        sample_path = f"{base}_sample{ext}"
        # s.request.real_json_file = output_json_path
        s.request.json_file = sample_path

    async def operator_qa_node(state: MainState) -> MainState:
        current_api_key = state.request.api_key
        
        if current_api_key and rag_service.api_key != current_api_key:
            log.info(f"同步 API Key 到 RAG Service...")
            rag_service.api_key = current_api_key
            rag_service._searcher = None 

        tm = get_tool_manager()
        
        agent = create_operator_qa_agent(
            tool_manager=tm,
            rag_service=rag_service,
            model_name=state.request.model or "gpt-4o",
            temperature=0,
            max_tokens=4096,
            parser_type="json",
            tool_mode="auto",
            message_history=shared_message_history,
        )
        
        state = await agent.execute(state, use_agent=True)
        
        result = state.agent_results.get("operator_qa", {})
        log.info(f"OperatorQA 执行结果: {result}")
        
        return state

    async def write_node(s: DFState) -> DFState:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        
        agent = create_writer(tool_manager=get_tool_manager(), model_name=_get_llm_model_name(s))
        return await agent.execute(s, use_agent=False)

    async def debugger_node(s: DFState) -> DFState:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager

        debugger = create_code_debugger(tool_manager=get_tool_manager(), model_name=_get_llm_model_name(s))
        return await debugger.execute(s, use_agent=True)

    async def rewriter_node(s: DFState) -> DFState:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager

        rewriter = create_rewriter(tool_manager=get_tool_manager(), model_name=_get_llm_model_name(s))
        return await rewriter.execute(s, use_agent=True)

    def after_rewrite_node(s: DFState) -> DFState:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager

        rewriter = create_rewriter(tool_manager=get_tool_manager(), model_name=_get_llm_model_name(s))
        return rewriter.after_rewrite(s)

    # ---------------- 新增：实例化节点（LLM 生成可运行入口 + 执行验证） --------
    async def instantiate_operator_main_node(s: DFState) -> DFState:
        out_s, err_s = "", ""
        returncode = -1

        agent = create_llm_instantiator(
            tool_manager=get_tool_manager(), 
            model_name=_get_llm_model_name(s)
        )
        s2 = await agent.execute(s, use_agent=True)
        
        code_str = s2.temp_data.get("pipeline_code", "") or getattr(s2, "draft_operator_code", "")
        if not code_str:
            log.warning("Agent 未生成任何代码，跳过执行环节。")
            return s2

        file_path = s2.temp_data.get("pipeline_file_path")
        if not file_path or not os.path.exists(file_path):
            log.error(f"代码文件不存在: {file_path}")
            return s2

        try:
            returncode, out_s, err_s = run_python_file(file_path)

            s2.temp_data.setdefault("debug_runtime", {})
            if returncode != 0:
                s2.temp_data["debug_runtime"]["exec_error"] = f"Runtime Error: Exit code {returncode}"
                s2.temp_data["debug_runtime"]["stack_trace"] = err_s
            else:
                s2.temp_data["debug_runtime"].pop("exec_error", None)

        except Exception as e:
            s2.temp_data.setdefault("debug_runtime", {})
            s2.temp_data["debug_runtime"]["exec_error"] = f"Internal Exec Exception: {str(e)}"
            s2.temp_data["debug_runtime"]["stack_trace"] = traceback.format_exc()

        exec_error = s2.temp_data.get("debug_runtime", {}).get("exec_error")
        success = (returncode == 0) and (not exec_error)

        s2.temp_data["debug_runtime"].update({
            "stdout": out_s[:2000] if out_s else "",
            "stderr": err_s[:2000] if err_s else "",
        })

        s2.execution_result = {
            "success": success,
            "stdout": out_s,
            "stderr": err_s or exec_error or "",
            "file_path": file_path,
            "exec_error": exec_error or "",
            "stack_trace": s2.temp_data["debug_runtime"].get("stack_trace", ""),
        }
        return s2

    def exec_condition(s: DFState):
        if s.request.need_debug:
            if s.execution_result.get("success"):
                return "__end__"
            if s.temp_data.get("round", 0) >= s.request.max_debug_rounds:
                return "__end__"
            return "filter_code_debugger"
        else:
            return "__end__"

    nodes = {
        "preprocess_data_node": preprocess_data_node,
        "operator_qa_node": operator_qa_node,
        "write_filter_pipeline": write_node,
        "filter_llm_instantiate": instantiate_operator_main_node,
        "filter_code_debugger": debugger_node,
        "filter_rewriter": rewriter_node,
        "after_rewrite": after_rewrite_node,
    }
    
    edges = [
        ("preprocess_data_node", "operator_qa_node"),
        ("operator_qa_node", "write_filter_pipeline"),
        ("write_filter_pipeline", "filter_llm_instantiate"),
        ("filter_code_debugger", "filter_rewriter"),
        ("filter_rewriter", "after_rewrite"),
        ("after_rewrite", "filter_llm_instantiate"),
    ]

    builder.add_nodes(nodes, role_mapping={"operator_qa_node": "operator_qa"}).add_edges(edges).add_conditional_edges({"filter_llm_instantiate": exec_condition})
    return builder