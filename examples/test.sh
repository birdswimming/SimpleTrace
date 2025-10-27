#!/usr/bin/env bash
export TINYTRACE_LOG_LEVEL="debug"
export TINYTRACE_LEVEL="info"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MLP_TASK_ID="${MLP_TASK_ID:-t-$(date +%Y%m%d%H%M%S)-local}"
export WORK_DIR="test_trace/$MLP_TASK_ID/"
mkdir -p "test_trace"
mkdir ${WORK_DIR}

python "${SCRIPT_DIR}/test.py"