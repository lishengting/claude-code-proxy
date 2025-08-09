#!/usr/bin/env bash

# [删除替换-原因] 直接使用 'set -euo pipefail' 在 /bin/sh 下会报 "Illegal option -o pipefail"；改为在 bash 下才开启 pipefail，以兼容 'sh start.sh'
# set -euo pipefail
set -eu
if [ -n "${BASH-}" ]; then
  set -o pipefail
fi

ENV_FILE="${1:-.env}"

echo "[start.sh] 使用环境文件: ${ENV_FILE} (若不存在则忽略)"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  echo "[start.sh] 已加载环境变量自 ${ENV_FILE}"
else
  echo "[start.sh] 未找到 ${ENV_FILE}，将直接启动，不加载额外环境变量"
fi

echo "[start.sh] 启动代理: python start_proxy.py"
exec python start_proxy.py


