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
modal run scripts/scenesmith_modal.py \
  --prompt "A modern kitchen with an island and cluttered countertops." \
  --mode room
```

For explicit Hydra overrides:

```bash
modal run scripts/scenesmith_modal.py \
  --prompt "A studio apartment with a bed and desk." \
  --extra-overrides "experiment.pipeline.stop_stage=furniture"
```

## 4. Retrieve outputs

The default output volume name is `scenesmith-outputs`:

```bash
modal volume ls scenesmith-outputs /
modal volume get scenesmith-outputs modal_YYYYMMDD_HHMMSS ./modal-output
```

## Notes

- The Modal path forces `SCENESMITH_DISABLE_BWRAP=1`, because Modal does not provide the container capabilities used by docker-compose for bubblewrap GPU isolation.
- The script defaults to `gpu="L40S"`. Override with `SCENESMITH_MODAL_GPU=A100-80GB` if needed.
- Volume and secret names can be overridden with `SCENESMITH_MODAL_*` environment variables before running `modal run`.
