import importlib
from pathlib import Path

def register_all_workflows():
    _pkg_path = Path(__file__).resolve().parent
    # 动态导入 wf_ 开头的文件，触发里面的 @workflow.register 装饰器
    for py in _pkg_path.glob("wf_*.py"):
        mod_name = f"{__package__}.{py.stem}"
        importlib.import_module(mod_name)