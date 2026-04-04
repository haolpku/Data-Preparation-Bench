"""执行时间记录工具."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager


class TimingCollector:
    """收集执行时间的单例类."""

    _instance: TimingCollector | None = None

    def __new__(cls) -> TimingCollector:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._timings: dict[str, float] = {}
            self._enabled = True
            self._initialized = True

    def reset(self) -> None:
        """重置所有计时."""
        self._timings.clear()

    def record(self, name: str, duration: float) -> None:
        """记录一个时间."""
        if not self._enabled:
            return
        if name in self._timings:
            # 如果已存在，累加时间（支持多次执行）
            self._timings[name] += duration
        else:
            self._timings[name] = duration

    def get_all(self) -> dict[str, float]:
        """获取所有记录的时间."""
        return self._timings.copy()

    def get_total(self) -> float:
        """获取总时间."""
        return sum(self._timings.values())

    def set_enabled(self, enabled: bool) -> None:
        """启用或禁用计时."""
        self._enabled = enabled

    def format_report(self) -> str:
        """格式化时间报告."""
        if not self._timings:
            return "暂无计时记录"

        lines = ["\n" + "=" * 60, "执行时间统计", "=" * 60]

        # 按时间排序
        sorted_timings = sorted(self._timings.items(), key=lambda x: x[1], reverse=True)

        max_name_len = max(len(name) for name in self._timings.keys())

        for name, duration in sorted_timings:
            lines.append(f"  {name:<{max_name_len}} : {duration:>8.3f}s")

        lines.append("-" * 60)
        lines.append(f"  {'总计':<{max_name_len}} : {self.get_total():>8.3f}s")
        lines.append("=" * 60)

        return "\n".join(lines)


# 全局实例
timing_collector = TimingCollector()


@contextmanager
def timing_context(name: str) -> Generator[None]:
    """计时上下文管理器.

    使用方式:
        with timing_context("数据加载"):
            load_dataset(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        timing_collector.record(name, duration)


def reset_timing() -> None:
    """重置计时器."""
    timing_collector.reset()


def get_timing_report() -> str:
    """获取计时报告."""
    return timing_collector.format_report()


def get_timings() -> dict[str, float]:
    """获取所有计时数据."""
    return timing_collector.get_all()
