"""日志工具 - 支持彩色输出、函数信息和分级."""

import inspect
import logging
import sys
from typing import Any


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器."""

    # ANSI 颜色代码
    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",  # 重置
    }

    def format(self, record: logging.LogRecord) -> str:
        # 获取颜色
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # 给级别名称添加颜色
        record.levelname = f"{color}{levelname}{reset}"

        return super().format(record)


def _get_caller_info() -> str:
    """获取调用者信息 (模块名.函数名)."""
    # 获取调用栈，跳过当前函数和logging内部函数
    frame = inspect.currentframe()
    if frame is None:
        return "unknown"

    try:
        # 向上查找调用者 (跳过logger.py内部和logging框架)
        caller_frame = frame
        for _ in range(3):  # 跳过 _get_caller_info -> log function -> logger call
            if caller_frame.f_back is None:
                break
            caller_frame = caller_frame.f_back

        module = inspect.getmodule(caller_frame)
        module_name = module.__name__ if module else "unknown"
        function_name = caller_frame.f_code.co_name

        return f"{module_name}.{function_name}"
    finally:
        del frame


def _create_logger() -> logging.Logger:
    """创建并配置 logger."""
    logger = logging.getLogger("mmd")
    logger.setLevel(logging.DEBUG)

    # 如果已经有 handler，不再添加
    if logger.handlers:
        return logger

    # 创建控制台 handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # 设置格式化器
    formatter = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# 全局 logger 实例
_logger = _create_logger()


def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """输出 DEBUG 级别日志."""
    caller_info = _get_caller_info()
    _logger.debug(f"[{caller_info}] {msg}", *args, **kwargs)


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    """输出 INFO 级别日志."""
    caller_info = _get_caller_info()
    _logger.info(f"[{caller_info}] {msg}", *args, **kwargs)


def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """输出 WARNING 级别日志."""
    caller_info = _get_caller_info()
    _logger.warning(f"[{caller_info}] {msg}", *args, **kwargs)


def error(msg: str, *args: Any, **kwargs: Any) -> None:
    """输出 ERROR 级别日志."""
    caller_info = _get_caller_info()
    _logger.error(f"[{caller_info}] {msg}", *args, **kwargs)


def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    """输出 CRITICAL 级别日志."""
    caller_info = _get_caller_info()
    _logger.critical(f"[{caller_info}] {msg}", *args, **kwargs)


def set_level(level: int | str) -> None:
    """设置日志级别.

    Args:
        level: 日志级别，可以是 logging.DEBUG, logging.INFO 等，
               或字符串 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    _logger.setLevel(level)
