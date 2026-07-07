# vllm-docker-optimized (spark-vllm-docker)

## Purpose
Serve LLMs using Docker containers running vLLM on one or more **NVIDIA DGX Spark (GB10, 128GB unified memory)** hosts — single-node ("solo") or across a discovered 2x/4x Spark cluster (Ray/InfiniBand). Targets Blackwell `TORCH_CUDA_ARCH_LIST=12.1a`.

## Architecture (layered scripts — no docker-compose, no Makefile)
- `run-recipe.sh` → `run-recipe.py` (the brain: parses recipe YAML, `{param}` substitution, `--setup` build+download) → `launch-cluster.sh` (Docker orchestrator: `docker run … sleep infinity`, applies mods, injects `vllm serve` via `docker exec`).
- `build-and-copy.sh` (build + scp image to peers), `hf-download.sh` (download + rsync model), `autodiscover.sh` (cluster discovery → `.env`).
- Default image is pulled prebuilt: `eugr/spark-vllm:latest`.

## Recipes & mods
- `recipes/*.yaml`: declarative model configs (`name`, `container`, `command` template, `defaults:` {port, gpu_memory_utilization, …}, `env:`, `mods:`). Format spec in `recipes/README.md`.
- Mods (`mods/<name>/run.sh`) are applied **at runtime** inside the running container (docker cp → run.sh). Category A patches the vLLM install (`git apply` into `/usr/local/lib/python3.12/dist-packages`); Category B drops chat-template `.jinja` files. Each container has its own vLLM install, so per-recipe patches are isolated.

## Stacks (co-locate multiple recipes on one host)
- `run-stack.sh <stack>` runs several recipes at once (each own container/port), **solo-only**. Manifests in `stacks/*.yaml`; spec in `stacks/README.md`.
- Recipes launch in listed order (**must be descending memory usage**), health-gated on `/health` before the next starts. `--dry-run`/`--status`/`--stop`/`--setup`/`--list`.

## Common commands
- `./run-recipe.sh --list` · `./run-recipe.sh <recipe> --solo [--setup]` · `./run-recipe.sh <recipe> -n ip1,ip2 --setup` (cluster)
- `./run-stack.sh <stack> --dry-run|--setup|--status|--stop`
- Override flags: `--port`, `--gpu-mem`, `--tp`, `--max-model-len`; extra vLLM args go **after `--`**.

## Testing (bash, dry-run based — no pytest)
- `./tests/test_recipes.sh -v` and `./tests/test_stack.sh -v`; CI is `.github/workflows/test-recipes.yml` (matrix py3.10–3.12). Only Python dep is PyYAML (wrappers auto-install it).

## PR conventions (drafting)
Reference: [#310](https://github.com/eugr/spark-vllm-docker/pull/310) "feat: honor HF_HUB_CACHE and propagate HF_HUB_OFFLINE" — extends `HF_HOME` (#68) so the HF cache resolves `HF_HUB_CACHE` → `$HF_HOME/hub` → `~/.cache/huggingface/hub` (touches `hf-download.sh` HUB_PATH, `run-recipe.py` `check_model_exists()`, and `launch-cluster.sh` hub bind-mount + `HF_HUB_OFFLINE` propagation).
- PR body structure: `## Summary` (motivating use case + cross-reference related issues/PRs by number), a per-file `Changes:` bullet list, `Notes / known limitations`, then `## Test plan` with `[x]` checkboxes.
- Test-plan norms: run `./tests/test_recipes.sh` and cite the pass count (e.g. 56/56); `bash -n` + `py_compile` clean on changed scripts; verify the env/flag resolution matrix; prove **no regression** by showing generated `docker run` args are byte-identical to `main` when new vars are unset; include a real-world DGX Spark validation.

## Gotchas
- Put `run-recipe.py`'s own flags (e.g. `--dry-run`) **before** the `--` separator; everything after `--` is passed through to vLLM.
- `--gpu-memory-utilization` is a fraction of **total** memory; the `mods/gpu-mem-util-gb` mod adds absolute-GiB (`--gpu-memory-utilization-gb`), which composes better when stacking models.
- Containers run `sleep infinity` with the image entrypoint cleared; the actual command is injected via `docker exec` (daemon mode = `docker exec -d`).
- Start new, unrelated features on a fresh branch cut from `main` (not stacked on an open PR branch).
