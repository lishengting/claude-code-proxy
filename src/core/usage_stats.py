import os
import threading
from datetime import datetime
from typing import Any, Dict, List

from src.core.config import config
from src.core.logging import logger

_TSV_HEADER_COLUMNS: List[str] = [
    "timestamp",
    "request_id",
    "is_stream",
    "model",
    "base_url",
    "api_type",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "total_tokens",
    "latency_ms",
    "status",
    "error",
]

_file_lock = threading.Lock()


def _ensure_parent_dir_exists(filepath: str) -> None:
    parent_dir = os.path.dirname(os.path.abspath(filepath))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


def append_usage_tsv(record: Dict[str, Any]) -> None:
    """将用量统计以TSV格式追加写入统计文件。

    必填字段见 _TSV_HEADER_COLUMNS；缺失字段将按空字符串处理。
    """
    # 进入日志
    logger.debug(
        f"[USAGE] 进入 append_usage_tsv: model={record.get('model')}, stream={record.get('is_stream')}, in={record.get('input_tokens', 0)}, out={record.get('output_tokens', 0)}"
    )
    try:
        filepath = getattr(config, "usage_stats_path", "./openai_usage.tsv")
        _ensure_parent_dir_exists(filepath)

        # 若文件不存在则写入表头
        file_exists = os.path.exists(filepath)
        linestr_values = []
        for key in _TSV_HEADER_COLUMNS:
            value = record.get(key, "")
            # 将换行和制表符替换，避免破坏TSV结构
            if value is None:
                value = ""
            value_str = str(value).replace("\t", " ").replace("\n", " ")
            linestr_values.append(value_str)
        line = "\t".join(linestr_values) + "\n"

        with _file_lock:
            with open(filepath, "a", encoding="utf-8") as f:
                if not file_exists:
                    header = "\t".join(_TSV_HEADER_COLUMNS) + "\n"
                    f.write(header)
                f.write(line)

        # 成功日志
        logger.debug(
            f"[USAGE] 写入统计成功: file={filepath}, model={record.get('model')}, request_id={record.get('request_id')}"
        )
    except Exception as e:
        # 失败日志
        logger.error(f"[USAGE] 写入统计失败: {e}")


