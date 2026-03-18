# Running SceneSmith on Modal

This repository now includes `scripts/scenesmith_modal.py`, which builds from `Dockerfile.modal` and runs `python main.py` inside a single Modal GPU container.

## 1. Create the Modal secret

Install the local Modal CLI/package, then create one secret containing the runtime keys:

```bash
modal secret create scenesmith-runtime \
  OPENAI_API_KEY=... \
  HF_TOKEN=...
```

Optional keys in the same secret:

```bash
GOOGLE_API_KEY=...
OPENAI_TRACING_KEY=...
```

## 2. Populate Modal volumes

This downloads the default runtime assets into Modal Volumes:

```bash
modal run scripts/scenesmith_modal.py::prepare_data --material-limit 100
```

`material-limit=100` keeps the first run smaller. Pass `0` to fetch the full AmbientCG library.

## 3. Generate a scene

```bash
uv run modal run scripts/scenesmith_modal.py \
  --prompt "A modern kitchen with an island and cluttered countertops." \
  --mode room
```

The Modal wrapper now defaults to a 24-hour function timeout. Override it if needed:

```bash
SCENESMITH_MODAL_GENERATE_TIMEOUT_S=43200 \
uv run modal run scripts/scenesmith_modal.py \
  --prompt "A studio apartment with a bed and desk."
```

## 4. Split long runs by pipeline stage

If a full run takes too long, stop after an expensive checkpointed stage and resume later.

First pass, stop after furniture:

```bash
uv run modal run scripts/scenesmith_modal.py \
  --prompt "A studio apartment with a bed and desk." \
  --run-name studio_base \
  --stop-stage furniture
```

Later, branch from that saved run and continue from wall-mounted objects:

```bash
uv run modal run scripts/scenesmith_modal.py \
  --prompt "A studio apartment with a bed and desk." \
  --run-name studio_final \
  --start-stage wall_mounted \
  --resume-run-id studio_base
```

Use `--start-stage ceiling_mounted` or `--start-stage manipuland` to rerun only later stages. For raw Hydra overrides, `--extra-overrides` is still passed through unchanged.

To continue writing into the same output directory after a timeout, rerun with the original `--run-name` and a later `--start-stage`.

## 5. Retrieve outputs

The default output volume name is `scenesmith-outputs`:

```bash
uv run modal volume ls scenesmith-outputs /
uv run modal volume get scenesmith-outputs modal_YYYYMMDD_HHMMSS ./modal-output
```

## Notes

- The Modal path forces `SCENESMITH_DISABLE_BWRAP=1`, because Modal does not provide the container capabilities used by docker-compose for bubblewrap GPU isolation.
- The script defaults to `gpu="L40S"`. Override with `SCENESMITH_MODAL_GPU=A100-80GB` if needed.
- Volume and secret names can be overridden with `SCENESMITH_MODAL_*` environment variables before running `modal run`.
- Modal Function timeouts currently max out at 24 hours; longer jobs should be made resumable with stage splits and the output volume. Source: https://modal.com/docs/examples/long-training
