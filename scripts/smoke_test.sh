#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/Qwen3.8-27B-AWQ}"
python - "$OUTPUT_DIR" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoConfig, AutoProcessor

path = Path(sys.argv[1])
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
AutoProcessor.from_pretrained(path, trust_remote_code=True)
quant = getattr(config, "quantization_config", None)
if not quant:
    raise SystemExit("missing quantization_config")
print(json.dumps(quant, indent=2, default=str))
print("checkpoint-smoke=ok")
PY

