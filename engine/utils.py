"""engine.utils

项目通用工程工具函数。
"""

from __future__ import annotations

import shutil
from pathlib import Path


def has_sufficient_disk_space(path: str | Path, min_gb: float = 2.0) -> bool:
    """检查指定路径所在的磁盘是否有足够的剩余空间（单位 GB）。

    参数:
        path: 待检查的目录或文件路径。
        min_gb: 要求的最小剩余空间（GB）。

    返回:
        bool: True 表示空间足够或检查失败（保守尝试），False 表示空间明确不足。
    """
    try:
        # 转换为绝对路径并获取其所在目录
        p = Path(path).resolve()
        # 如果路径不存在，检查其父目录
        check_path = p if p.exists() else p.parent
        while not check_path.exists() and check_path != check_path.parent:
            check_path = check_path.parent

        total, used, free = shutil.disk_usage(check_path)
        free_gb = free / (1024**3)
        return free_gb >= min_gb
    except Exception:
        # 如果获取失败（如权限问题或特殊文件系统），保守起见返回 True 允许尝试写入
        return True
