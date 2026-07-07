#!/bin/bash
#
# test_stack.sh - Tests for run-stack.py (multi-recipe co-location).
#
# Uses --dry-run and validation paths only; never launches containers.
# Suitable for CI.
#
# Usage:
#   ./tests/test_stack.sh        # run all tests
#   ./tests/test_stack.sh -v     # verbose
#

set +e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_STACK="$PROJECT_DIR/run-stack.py"
VERBOSE="${1:-}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
TESTS_PASSED=0; TESTS_FAILED=0

log_test() { echo -e "${YELLOW}[TEST]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; TESTS_PASSED=$((TESTS_PASSED + 1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; TESTS_FAILED=$((TESTS_FAILED + 1)); }
log_verbose() { [[ "$VERBOSE" == "-v" ]] && echo "       $1"; return 0; }

# assert_contains <description> <haystack> <needle>
assert_contains() {
    if [[ "$2" == *"$3"* ]]; then
        log_pass "$1"
    else
        log_fail "$1"
        log_verbose "expected to find: $3"
        log_verbose "in output:"
        log_verbose "$2"
    fi
}

TMPDIR_STACK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_STACK"' EXIT

echo "========================================"
echo "run-stack.py tests"
echo "========================================"

# ---- 1. --list surfaces the example stack ----
log_test "--list includes example-dual"
OUT="$("$RUN_STACK" --list 2>&1)"
assert_contains "example-dual listed" "$OUT" "example-dual"

# ---- 2. example-dual dry-run: order, names, ports ----
log_test "example-dual --dry-run launch plan"
OUT="$("$RUN_STACK" example-dual --dry-run 2>&1)"
log_verbose "$OUT"
assert_contains "entry 1 is qwen (largest first)" "$OUT" \
    "1. ./run-recipe.py qwen3.6-35b-a3b-nvfp4 --solo --name qwen_nvfp4 -d --port 8000"
assert_contains "entry 2 is nemotron" "$OUT" \
    "2. ./run-recipe.py nemotron-3-nano-nvfp4 --solo --name nemotron_nano -d --port 8001"
assert_contains "health gate on port 8000" "$OUT" "http://localhost:8000/health"
assert_contains "health gate on port 8001" "$OUT" "http://localhost:8001/health"
# Order check: qwen must appear before nemotron in the output.
if [[ "$OUT" == *"qwen_nvfp4"*"nemotron_nano"* ]]; then
    log_pass "load order preserved (qwen before nemotron)"
else
    log_fail "load order preserved (qwen before nemotron)"
fi

# ---- 3. duplicate port aborts before launching ----
log_test "duplicate port is rejected"
cat > "$TMPDIR_STACK/dup-port.yaml" <<'EOF'
name: dup-port
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
  - {recipe: nemotron-3-nano-nvfp4, container_name: b, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/dup-port.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"duplicate port"* ]]; then
    log_pass "duplicate port errors with nonzero exit"
else
    log_fail "duplicate port errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 4. duplicate container_name aborts ----
log_test "duplicate container_name is rejected"
cat > "$TMPDIR_STACK/dup-name.yaml" <<'EOF'
name: dup-name
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
  - {recipe: nemotron-3-nano-nvfp4, container_name: a, port: 8001}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/dup-name.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"duplicate container_name"* ]]; then
    log_pass "duplicate container_name errors with nonzero exit"
else
    log_fail "duplicate container_name errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 5. port defaults from the recipe when omitted ----
log_test "omitted port inherits recipe defaults.port"
cat > "$TMPDIR_STACK/default-port.yaml" <<'EOF'
name: default-port
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: only}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/default-port.yaml" --dry-run 2>&1)"
# qwen3.6-35b-a3b-nvfp4 recipe declares defaults.port: 8000
assert_contains "default port 8000 applied" "$OUT" "--port 8000"

# ---- 6. mods resolve to absolute paths; extra_args land after `--` ----
log_test "mods + extra_args rendering"
cat > "$TMPDIR_STACK/mods.yaml" <<'EOF'
name: mods
recipes:
  - recipe: qwen3.6-35b-a3b-nvfp4
    container_name: q
    port: 8000
    mods: [mods/gpu-mem-util-gb]
    extra_args: ["--gpu-memory-utilization-gb", 60]
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/mods.yaml" --dry-run 2>&1)"
log_verbose "$OUT"
assert_contains "mod resolved to absolute path" "$OUT" \
    "--apply-mod $PROJECT_DIR/mods/gpu-mem-util-gb"
assert_contains "extra_args after --" "$OUT" \
    "-- --gpu-memory-utilization-gb 60"

# ---- 7. missing recipe field is rejected ----
log_test "entry without 'recipe' is rejected"
cat > "$TMPDIR_STACK/no-recipe.yaml" <<'EOF'
name: no-recipe
recipes:
  - {container_name: a, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/no-recipe.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"must have a 'recipe'"* ]]; then
    log_pass "missing recipe field errors"
else
    log_fail "missing recipe field errors (rc=$RC)"
    log_verbose "$OUT"
fi

echo "========================================"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}  Failed: ${RED}${TESTS_FAILED}${NC}"
echo "========================================"
[[ $TESTS_FAILED -eq 0 ]]
