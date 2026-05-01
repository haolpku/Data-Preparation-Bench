from typing import Any, Dict, List
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage

from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import MainState
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.logger import get_logger
# from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

class FilterBaseAgent(BaseAgent):
    def build_messages(self, 
                       state: MainState, 
                       pre_tool_results: Dict[str, Any]) -> List[BaseMessage]:
        """
        构建 LLM 输入消息列表
        
        根据系统提示词模板和任务提示词模板生成完整的消息列表，
        包括格式说明（如果使用解析器）。
        
        Args:
            state (MainState): 当前状态对象，包含请求信息
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            List[BaseMessage]: 消息列表，包含 SystemMessage 和 HumanMessage
        
        消息结构：
            1. SystemMessage: 系统提示词 + 格式说明
            2. HumanMessage: 任务提示词（包含前置工具结果）
        """
        log.info("构建提示词消息...")
        
        # 创建提示词生成器
        ptg = PromptsTemplateGenerator(
            state.request.language,
            template_dirs=[f'{get_project_root()}/promptstemplates']
        )
        
        # 渲染系统提示词
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        # 添加解析器格式说明（VLM 模式可能不需要）
        format_instruction = self.parser.get_format_instruction()
        if format_instruction and not self.use_vlm:
            sys_prompt += f"\n\n{format_instruction}"
        
        # 渲染任务提示词
        task_params = self.get_task_prompt_params(pre_tool_results)
        task_prompt = ptg.render(self.task_prompt_template_name, **task_params)
        log.info(f"[build_messages]任务提示词: {task_prompt}")
        
        # 构建消息列表
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]
        
        log.info("提示词消息构建完成")
        return messages