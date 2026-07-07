#!/usr/bin/env python3
"""
run-stack.py - Co-locate multiple vLLM recipes on a single host.

A "stack" is a declarative YAML manifest (see stacks/*.yaml) listing several
recipes to serve at once on one machine, each in its own Docker container on its
own port. This driver is a thin layer on top of run-recipe.py: it launches each
recipe in the listed order (which MUST be descending memory usage) and waits for
each recipe's /health endpoint to report ready before starting the next one.

Why the order and the health gate matter: vLLM profiles free memory at startup,
so the largest model must claim its slice against a clean pool first (and fail
fast if it won't fit). Health-gating each launch also serializes first-time
kernel compilation so co-located instances don't race on the shared compile
caches.

Scope: v1 is solo (single host) only. Cluster co-location (running several
cluster-spanning recipes at once) is a documented follow-up; the manifest and
this driver are shaped so it slots in without a redesign.

Usage:
    ./run-stack.sh <stack> [--setup] [--dry-run]   # bring the stack up
    ./run-stack.sh <stack> --status                # show each container + health
    ./run-stack.sh <stack> --stop                  # tear the whole stack down
"""

import argparse
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_RECIPE = SCRIPT_DIR / "run-recipe.py"
LAUNCH_CLUSTER = SCRIPT_DIR / "launch-cluster.sh"
STACKS_DIR = SCRIPT_DIR / "stacks"
RECIPES_DIR = SCRIPT_DIR / "recipes"

DEFAULT_HEALTH_TIMEOUT = 1200  # seconds to wait for a recipe to become ready
HEALTH_POLL_INTERVAL = 5       # seconds between /health polls
DEFAULT_PORT = 8000
STACK_VERSIONS = ("1",)        # manifest versions this driver understands

# Docker's accepted container-name charset.
CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


# --------------------------------------------------------------------------- #
# Manifest loading / resolution
# --------------------------------------------------------------------------- #
def resolve_stack_path(name_or_path):
    """Resolve a stack manifest by path or by name under stacks/."""
    candidates = [
        Path(name_or_path),
        STACKS_DIR / name_or_path,
        STACKS_DIR / f"{name_or_path}.yaml",
        STACKS_DIR / f"{name_or_path}.yml",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"Stack manifest '{name_or_path}' not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def resolve_recipe_path(recipe):
    """Resolve a recipe by path or name under recipes/ (mirrors run-recipe.py)."""
    candidates = [
        Path(recipe),
        RECIPES_DIR / recipe,
        RECIPES_DIR / f"{recipe}.yaml",
        RECIPES_DIR / f"{recipe}.yml",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def recipe_default_port(recipe_path):
    """Read defaults.port from a resolved recipe, falling back to DEFAULT_PORT."""
    try:
        data = yaml.safe_load(recipe_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return DEFAULT_PORT
    return (data.get("defaults") or {}).get("port", DEFAULT_PORT)


def sanitize_name(recipe):
    """Derive a Docker-safe default container name from a recipe name."""
    base = Path(recipe).name
    # Strip only a manifest extension: Path.stem would truncate dotted recipe
    # names like 'qwen3.6-35b-a3b-nvfp4' to 'qwen3'.
    for ext in (".yaml", ".yml"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    cleaned = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in base)
    return cleaned or "vllm_node"


def load_stack(name_or_path):
    """Load and normalize a stack manifest into a list of resolved entries."""
    path = resolve_stack_path(name_or_path)
    data = yaml.safe_load(path.read_text()) or {}

    version = data.get("stack_version")
    if version is not None and str(version) not in STACK_VERSIONS:
        raise ValueError(
            f"Stack '{path}': unsupported stack_version {version!r} "
            f"(supported: {', '.join(STACK_VERSIONS)})."
        )
    if not data.get("solo_only", True):
        raise ValueError(
            f"Stack '{path}': 'solo_only: false' is not supported yet; "
            "cluster co-location is a planned follow-up."
        )

    raw_entries = data.get("recipes")
    if not raw_entries:
        raise ValueError(f"Stack '{path}' has no 'recipes:' list.")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Stack '{path}': 'recipes:' must be a list.")

    entries = []
    seen_names = {}
    seen_ports = {}
    for i, raw in enumerate(raw_entries):
        if isinstance(raw, str):
            raw = {"recipe": raw}
        if not isinstance(raw, dict) or not raw.get("recipe"):
            raise ValueError(
                f"Stack '{path}': entry #{i + 1} must have a 'recipe' field."
            )
        recipe = raw["recipe"]
        recipe_path = resolve_recipe_path(recipe)
        if recipe_path is None:
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: recipe '{recipe}' not found "
                f"(looked for a file at that path and under {RECIPES_DIR}/)."
            )
        explicit_name = raw.get("container_name")
        cname = explicit_name or sanitize_name(recipe)
        if not CONTAINER_NAME_RE.match(cname):
            derived = (
                "" if explicit_name
                else f" (derived from recipe '{recipe}'; set 'container_name' explicitly)"
            )
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: container_name '{cname}' is not "
                "a valid Docker container name "
                f"(must match [a-zA-Z0-9][a-zA-Z0-9_.-]*){derived}."
            )
        port = raw.get("port")
        if port is None:
            port = recipe_default_port(recipe_path)
        try:
            port = int(port)
        except (TypeError, ValueError):
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: 'port' must be an integer, "
                f"got {port!r}."
            ) from None

        if cname in seen_names:
            raise ValueError(
                f"Stack '{path}': duplicate container_name '{cname}' "
                f"(entries #{seen_names[cname] + 1} and #{i + 1}). "
                "Set an explicit 'container_name' for one of them."
            )
        if port in seen_ports:
            raise ValueError(
                f"Stack '{path}': duplicate port {port} "
                f"(entries #{seen_ports[port] + 1} and #{i + 1}). "
                "Each co-located recipe needs its own port."
            )
        seen_names[cname] = i
        seen_ports[port] = i

        # Resolve mod paths relative to the repo root so run-recipe.py's
        # cwd-relative resolution doesn't depend on where run-stack is invoked.
        mods = []
        for m in raw.get("mods", []) or []:
            mp = Path(m)
            mods.append(str(mp if mp.is_absolute() else (SCRIPT_DIR / mp)))

        entries.append(
            {
                "recipe": recipe,
                "container_name": cname,
                "port": port,
                "gpu_mem": raw.get("gpu_mem"),
                "mods": mods,
                "extra_args": [str(a) for a in (raw.get("extra_args") or [])],
            }
        )

    raw_timeout = data.get("health_timeout", DEFAULT_HEALTH_TIMEOUT)
    try:
        health_timeout = int(raw_timeout)
    except (TypeError, ValueError):
        raise ValueError(
            f"Stack '{path}': 'health_timeout' must be an integer (seconds), "
            f"got {raw_timeout!r}."
        ) from None

    return {
        "path": path,
        "name": data.get("name", path.stem),
        "health_timeout": health_timeout,
        "entries": entries,
    }


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
def recipe_args(entry, setup, daemon=True):
    """Args to run-recipe.py for one entry (everything after the program name)."""
    args = [entry["recipe"], "--solo", "--name", entry["container_name"]]
    if daemon:
        args.append("-d")
    args += ["--port", str(entry["port"])]
    if entry["gpu_mem"] is not None:
        args += ["--gpu-mem", str(entry["gpu_mem"])]
    for m in entry["mods"]:
        args += ["--apply-mod", m]
    if setup:
        args.append("--setup")
    if entry["extra_args"]:
        args += ["--", *entry["extra_args"]]
    return args


def health_url(host, port):
    return f"http://{host}:{port}/health"


def is_healthy(host, port, timeout=2):
    try:
        with urllib.request.urlopen(health_url(host, port), timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def container_state(cname):
    """Return the container's Docker state ('running', 'exited', ...) or None if absent."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", cname],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def wait_healthy(host, port, cname, timeout):
    """Poll /health until ready. Returns None on success, else a failure reason.

    Also watches the container itself so a crash during startup fails fast
    instead of waiting out the full timeout. (vLLM runs via `docker exec`, so a
    dead vLLM process can leave the container 'running'; this catches
    container-level death, the timeout catches the rest.)
    """
    deadline = time.monotonic() + timeout
    url = health_url(host, port)
    print(f"    waiting for {url} (timeout {timeout}s)...", flush=True)
    while time.monotonic() < deadline:
        if is_healthy(host, port):
            print("    ready.", flush=True)
            return None
        state = container_state(cname)
        if state != "running":
            return (
                f"container '{cname}' "
                + (f"is in state '{state}'" if state else "no longer exists")
                + " before becoming healthy"
            )
        time.sleep(HEALTH_POLL_INTERVAL)
    return f"did not become healthy within {timeout}s"


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #
def cmd_dry_run(stack, host, setup):
    print(f"=== Stack: {stack['name']} ({len(stack['entries'])} recipes) ===")
    print(f"Health timeout per recipe: {stack['health_timeout']}s")
    print("Load order (as listed == descending memory usage):")
    print()
    for i, entry in enumerate(stack["entries"], 1):
        cmd = ["./run-recipe.py", *recipe_args(entry, setup=setup)]
        print(f"{i}. {shlex.join(cmd)}")
        print(f"   then wait for {health_url(host, entry['port'])}")
        print()
    return 0


def cmd_up(stack, host, setup):
    print(f"=== Bringing up stack: {stack['name']} ===", flush=True)
    for i, entry in enumerate(stack["entries"], 1):
        cname = entry["container_name"]
        port = entry["port"]
        print(
            f"\n[{i}/{len(stack['entries'])}] {entry['recipe']} "
            f"-> container '{cname}', port {port}",
            flush=True,
        )
        # If something is already answering /health on this port: our own
        # running container means this entry is already up (re-`up` is a
        # no-op for it); anything else owns the port, and vLLM would fail to
        # bind (host networking) while the health gate saw the impostor as
        # "ready" — refuse up front.
        if is_healthy(host, port):
            if container_state(cname) == "running":
                print(
                    f"    '{cname}' is already up and healthy on port {port}; "
                    "skipping.",
                    flush=True,
                )
                continue
            print(
                f"\nError: port {port} is already serving /health but container "
                f"'{cname}' is not running — another process owns that port. "
                f"Recipes started before this one are left running.",
                file=sys.stderr,
            )
            return 1
        cmd = [sys.executable, str(RUN_RECIPE), *recipe_args(entry, setup=setup)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"\nError: launching '{entry['recipe']}' failed "
                f"(run-recipe.py exit {result.returncode}). "
                f"Recipes started before this one are left running.",
                file=sys.stderr,
            )
            return result.returncode
        failure = wait_healthy(host, port, cname, stack["health_timeout"])
        if failure:
            print(
                f"\nError: '{entry['recipe']}' (container '{cname}', port {port}) "
                f"{failure}. Check logs: docker logs {cname}. "
                f"Recipes started before this one are left running.",
                file=sys.stderr,
            )
            return 1
    print(f"\n=== Stack '{stack['name']}' is up. ===")
    for entry in stack["entries"]:
        print(f"  {entry['recipe']:40s} http://{host}:{entry['port']}")
    return 0


def cmd_stop(stack):
    print(f"=== Stopping stack: {stack['name']} ===")
    rc = 0
    # Stop in reverse (last-started first) as a courtesy; order is not critical.
    for entry in reversed(stack["entries"]):
        cname = entry["container_name"]
        print(f"  stopping '{cname}'...")
        result = subprocess.run(
            [str(LAUNCH_CLUSTER), "--solo", "--name", cname, "stop"]
        )
        if result.returncode != 0:
            rc = result.returncode
    return rc


def cmd_status(stack, host):
    print(f"=== Stack: {stack['name']} ===")
    header = f"{'RECIPE':40s} {'CONTAINER':20s} {'PORT':>6s} {'STATE':>10s} {'HEALTH':>8s}"
    print(header)
    print("-" * len(header))
    for entry in stack["entries"]:
        cname = entry["container_name"]
        port = entry["port"]
        state_str = container_state(cname) or "absent"
        health = "ok" if is_healthy(host, port) else "-"
        print(
            f"{entry['recipe']:40s} {cname:20s} {port:>6d} "
            f"{state_str:>10s} {health:>8s}"
        )
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Co-locate multiple vLLM recipes on one host (solo).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stack", nargs="?", help="Stack manifest: name under stacks/ or a path"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available stacks"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Pass --setup to each recipe (build image + download model if missing)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to poll for /health (default: localhost)",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered run-recipe.py commands without executing",
    )
    action.add_argument("--stop", action="store_true", help="Stop the whole stack")
    action.add_argument(
        "--status", action="store_true", help="Show each container's state + health"
    )
    args = parser.parse_args()

    if args.list:
        if not STACKS_DIR.is_dir():
            print("No stacks/ directory found.")
            return 0
        stacks = sorted(
            p.stem for p in STACKS_DIR.glob("*.y*ml") if p.suffix in (".yaml", ".yml")
        )
        print("Available stacks:")
        for s in stacks:
            print(f"  {s}")
        return 0

    if not args.stack:
        parser.error("a stack name/path is required (or use --list)")

    try:
        stack = load_stack(args.stack)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.stop:
        return cmd_stop(stack)
    if args.status:
        return cmd_status(stack, args.host)
    if args.dry_run:
        return cmd_dry_run(stack, args.host, args.setup)
    return cmd_up(stack, args.host, args.setup)


if __name__ == "__main__":
    sys.exit(main())
