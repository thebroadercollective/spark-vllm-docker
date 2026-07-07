# Stacks — co-locating multiple recipes on one host

A **stack** runs several vLLM recipes at once on a single machine, each in its
own Docker container on its own port, sharing the same on-disk model cache. Use
it when two or more models each fit comfortably in the host's memory and you want
them served side by side (e.g. a large chat model on `:8000` and a small
tool/coder model on `:8001`).

Stacks are a thin layer over the normal recipe flow: `run-stack.py` calls
`run-recipe.py` once per recipe, so everything recipes already do — mods, model
download, container image selection, command templating — works unchanged.

> **Scope:** solo (single host) only for now. Running several *cluster-spanning*
> recipes at once is a planned follow-up; the manifest format is designed to
> extend to it (per-recipe `master_port` / Ray ports) without changing shape.

## Quick start

```bash
./run-stack.sh --list                     # list stacks/*.yaml
./run-stack.sh example-dual --dry-run     # show the ordered launch plan
./run-stack.sh example-dual --setup       # build/download as needed, then bring up
./run-stack.sh example-dual --status      # per-container state + /health
./run-stack.sh example-dual --stop        # tear the whole stack down
```

## Why order matters (and why it's sequential)

Recipes are launched **in the order listed**, and that order **must be
descending memory usage**. vLLM profiles free memory at startup, so the largest
model must claim its slice of the unified memory pool first — and fail fast if it
won't fit. Each recipe is started only after the previous one's `/health`
endpoint reports ready (`health_timeout` seconds, default 1200). This also
serializes first-time kernel compilation so co-located instances don't race on
the shared compile caches (`~/.cache/vllm`, `flashinfer`, `triton`, `tilelang`).

## Manifest format

```yaml
stack_version: "1"
name: my-stack                 # optional; defaults to the file name
description: ...               # optional
solo_only: true               # v1 is solo-only
health_timeout: 1200          # optional, seconds per recipe

recipes:                      # required; listed in DESCENDING memory order
  - recipe: <name-or-path>    # required; resolved like run-recipe.py (recipes/<name>.yaml)
    container_name: <name>    # optional; Docker --name, must be unique in the stack
                              #   (defaults to a sanitized recipe name)
    port: <int>               # optional; defaults to the recipe's defaults.port
    gpu_mem: <float>          # optional; fraction override -> run-recipe.py --gpu-mem
    mods: [<path>, ...]       # optional; extra --apply-mod paths (repo-relative)
    extra_args: [<arg>, ...]  # optional; appended after `--` to run-recipe.py (vLLM args)
```

The **served model name** (`--served-model-name`) and **port** stay recipe-owned
— `run-stack.py` doesn't invent them; it only forces the Docker `--name` to be
unique. `container_name`s and `port`s must each be unique across the stack, or
the run aborts with a clear error before anything launches.

### Memory budgeting: fractions vs absolute GiB

`gpu_mem` sets vLLM's `--gpu-memory-utilization` as a **fraction of total**
memory; the fractions across co-located recipes must sum to **< 1** with headroom
for CUDA context and non-torch allocations.

Absolute GiB budgets compose more predictably when stacking. The
[`mods/gpu-mem-util-gb`](../mods/gpu-mem-util-gb) mod patches vLLM to accept
`--gpu-memory-utilization-gb`; add it per entry and pass the flag via
`extra_args` (see the commented block in `example-dual.yaml`).

## Limitations when co-locating

- **Total memory** — the sum of all recipes (weights + KV cache + activations)
  must fit the host, with headroom. This is the main constraint on a GB10's
  128 GB unified pool.
- **Ray / distributed ports** — recipes using `--distributed-executor-backend
  ray` open Ray GCS/dashboard ports and a `MASTER_PORT` (29501). Two such recipes
  on host networking will collide. Single-GB10 solo recipes generally don't need
  this; it's the crux of the future cluster-co-location work.
- **Shared `/dev/shm` (`--ipc=host`)** — vLLM's engine↔API-server IPC uses shared
  memory; heavy co-location may need `--non-privileged` (per-container
  `--shm-size`), which the underlying recipe/launcher already supports.
- **One GPU** — compute is time-sliced across containers (no MIG on GB10);
  functionally fine, but throughput contends under simultaneous load.
- **`cluster_only` recipes** cannot be co-located in solo mode.
