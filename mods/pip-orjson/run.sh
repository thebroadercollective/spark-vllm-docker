#!/bin/bash
set -euo pipefail

# Installs the orjson Python package into the container's system environment.
# orjson accelerates JSON serialization and benefits several embedding models.
#
# Runs at launch time (before `vllm serve` is injected). Idempotent: skips the
# install if orjson already imports, so it is cheap to re-apply on every run.

PREFIX="[pip-orjson]"

if python3 -c "import orjson" >/dev/null 2>&1; then
  echo "$PREFIX orjson already present: $(python3 -c 'import orjson; print(orjson.__version__)')"
  exit 0
fi

echo "$PREFIX Installing orjson..."
uv pip install --system orjson
echo "$PREFIX Installed orjson $(python3 -c 'import orjson; print(orjson.__version__)')"
