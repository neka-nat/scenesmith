"""Modal entrypoint for running SceneSmith remotely.

Run from the repository root:

    modal run scripts/scenesmith_modal.py::prepare_data
    modal run scripts/scenesmith_modal.py --prepare --prompt "A modern kitchen..."

Required Modal secret (default name: ``scenesmith-runtime``):
    OPENAI_API_KEY
    HF_TOKEN

Optional keys in the same secret:
    GOOGLE_API_KEY
    OPENAI_TRACING_KEY
"""

from __future__ import annotations

import csv
import os
import shlex
import shutil
import subprocess

from datetime import datetime
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = Path("/app")
APP_BIN = APP_ROOT / ".venv" / "bin"
APP_PYTHON = APP_BIN / "python"
DATA_ROOT = APP_ROOT / "data"
CHECKPOINT_ROOT = APP_ROOT / "external" / "checkpoints"
OUTPUT_ROOT = APP_ROOT / "outputs"

APP_NAME = os.environ.get("SCENESMITH_MODAL_APP", "scenesmith")
SECRET_NAME = os.environ.get("SCENESMITH_MODAL_SECRET", "scenesmith-runtime")
DATA_VOLUME_NAME = os.environ.get("SCENESMITH_MODAL_DATA_VOLUME", "scenesmith-data")
CHECKPOINT_VOLUME_NAME = os.environ.get(
    "SCENESMITH_MODAL_CHECKPOINT_VOLUME", "scenesmith-checkpoints"
)
OUTPUT_VOLUME_NAME = os.environ.get(
    "SCENESMITH_MODAL_OUTPUT_VOLUME", "scenesmith-outputs"
)
DEFAULT_GPU = os.environ.get("SCENESMITH_MODAL_GPU", "L40S")
PIPELINE_STAGES = (
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
)


def _env_int(
    name: str,
    default: int,
    *,
    min_value: int = 1,
    max_value: int | None = None,
) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc

    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value


PREPARE_TIMEOUT_S = _env_int(
    "SCENESMITH_MODAL_PREPARE_TIMEOUT_S",
    3 * 60 * 60,
    min_value=10,
    max_value=24 * 60 * 60,
)
GENERATE_TIMEOUT_S = _env_int(
    "SCENESMITH_MODAL_GENERATE_TIMEOUT_S",
    24 * 60 * 60,
    min_value=10,
    max_value=24 * 60 * 60,
)
GENERATE_STARTUP_TIMEOUT_S = _env_int(
    "SCENESMITH_MODAL_GENERATE_STARTUP_TIMEOUT_S",
    60 * 60,
    min_value=10,
)

app = modal.App(APP_NAME)

image = modal.Image.from_dockerfile(str(REPO_ROOT / "Dockerfile.modal"))

runtime_secret = modal.Secret.from_name(SECRET_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env["PATH"] = f"{APP_BIN}:{merged_env.get('PATH', '')}"
    if env:
        merged_env.update(env)
    subprocess.run(
        cmd,
        check=True,
        cwd=APP_ROOT,
        env=merged_env,
    )


def _has_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _ensure_huggingface_cli() -> None:
    if (APP_BIN / "huggingface-cli").exists():
        return
    _run(
        [
            str(APP_PYTHON),
            "-m",
            "pip",
            "install",
            "huggingface_hub[cli]>=0.25.0",
        ]
    )


def _download_sam3d_checkpoints() -> None:
    _ensure_huggingface_cli()
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    sam3_path = CHECKPOINT_ROOT / "sam3.pt"
    pipeline_path = CHECKPOINT_ROOT / "pipeline.yaml"
    if sam3_path.exists() and pipeline_path.exists():
        return

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to download SAM3D checkpoints")

    if not sam3_path.exists():
        _run(
            [
                "huggingface-cli",
                "download",
                "facebook/sam3",
                "sam3.pt",
                "--local-dir",
                str(CHECKPOINT_ROOT),
            ]
        )

    if not pipeline_path.exists():
        tmp_root = Path("/tmp/sam3d-download")
        shutil.rmtree(tmp_root, ignore_errors=True)
        _run(
            [
                "huggingface-cli",
                "download",
                "facebook/sam-3d-objects",
                "--repo-type",
                "model",
                "--local-dir",
                str(tmp_root),
                "--include",
                "checkpoints/*",
            ]
        )
        checkpoints_dir = tmp_root / "checkpoints"
        for item in checkpoints_dir.iterdir():
            shutil.move(str(item), CHECKPOINT_ROOT / item.name)
        shutil.rmtree(tmp_root, ignore_errors=True)


def _download_artvip() -> None:
    artvip_root = DATA_ROOT / "artvip_sdf"
    if _has_contents(artvip_root):
        return

    _ensure_huggingface_cli()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to download ArtVIP data")

    download_root = Path("/tmp/scenesmith-artvip")
    shutil.rmtree(download_root, ignore_errors=True)
    _run(
        [
            "huggingface-cli",
            "download",
            "nepfaff/scenesmith-preprocessed-data",
            "artvip/artvip_vhacd.tar.gz",
            "--repo-type",
            "dataset",
            "--local-dir",
            str(download_root),
        ]
    )

    archive_path = download_root / "artvip" / "artvip_vhacd.tar.gz"
    artvip_root.mkdir(parents=True, exist_ok=True)
    _run(["tar", "xzf", str(archive_path), "-C", str(artvip_root)])
    shutil.rmtree(download_root, ignore_errors=True)


def _download_materials(material_limit: int) -> None:
    materials_root = DATA_ROOT / "materials"
    if _has_contents(materials_root):
        return

    cmd = [
        str(APP_PYTHON),
        "scripts/download_ambientcg.py",
        "--output",
        str(materials_root),
    ]
    if material_limit > 0:
        cmd.extend(["--limit", str(material_limit)])
    _run(cmd)


def _download_material_embeddings() -> None:
    embeddings_root = DATA_ROOT / "materials" / "embeddings"
    if _has_contents(embeddings_root):
        return

    _ensure_huggingface_cli()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to download AmbientCG embeddings")

    download_root = Path("/tmp/scenesmith-ambientcg")
    shutil.rmtree(download_root, ignore_errors=True)
    _run(
        [
            "huggingface-cli",
            "download",
            "nepfaff/scenesmith-preprocessed-data",
            "--repo-type",
            "dataset",
            "--include",
            "ambientcg/embeddings/**",
            "--local-dir",
            str(download_root),
        ]
    )

    source_root = download_root / "ambientcg" / "embeddings"
    embeddings_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_root), str(embeddings_root))
    shutil.rmtree(download_root, ignore_errors=True)


def _sanitize_run_name(run_name: str | None) -> str:
    if not run_name:
        return datetime.utcnow().strftime("modal_%Y%m%d_%H%M%S")
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name.strip()
    )
    return cleaned or datetime.utcnow().strftime("modal_%Y%m%d_%H%M%S")


def _runtime_env() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "0",
        "SCENESMITH_DISABLE_BWRAP": "1",
    }


def _validate_stage(stage: str, field_name: str) -> None:
    if stage not in PIPELINE_STAGES:
        raise ValueError(
            f"{field_name} must be one of {PIPELINE_STAGES}, got {stage!r}"
        )


def _resolve_resume_source(
    resume_from_path: str,
    resume_run_id: str,
) -> str | None:
    if resume_from_path and resume_run_id:
        raise ValueError("Use either resume_from_path or resume_run_id, not both")

    if resume_from_path:
        return resume_from_path.strip()

    if not resume_run_id.strip():
        return None

    run_id = resume_run_id.strip()
    if Path(run_id).name != run_id:
        raise ValueError("resume_run_id must be a run directory name, not a path")

    source_path = OUTPUT_ROOT / run_id
    if not source_path.exists():
        raise RuntimeError(f"Resume source does not exist: {source_path}")
    return str(source_path)


def _ensure_runtime_prereqs() -> None:
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not (CHECKPOINT_ROOT / "sam3.pt").exists():
        missing.append(str(CHECKPOINT_ROOT / "sam3.pt"))
    if not (CHECKPOINT_ROOT / "pipeline.yaml").exists():
        missing.append(str(CHECKPOINT_ROOT / "pipeline.yaml"))
    if not _has_contents(DATA_ROOT / "artvip_sdf"):
        missing.append(str(DATA_ROOT / "artvip_sdf"))
    if not _has_contents(DATA_ROOT / "materials"):
        missing.append(str(DATA_ROOT / "materials"))
    if not _has_contents(DATA_ROOT / "materials" / "embeddings"):
        missing.append(str(DATA_ROOT / "materials" / "embeddings"))

    if missing:
        raise RuntimeError(
            "SceneSmith Modal runtime is missing prerequisites: "
            + ", ".join(missing)
            + ". Run prepare_data first or populate the Modal volumes manually."
        )


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=PREPARE_TIMEOUT_S,
    secrets=[runtime_secret],
    volumes={
        str(DATA_ROOT): data_volume,
        str(CHECKPOINT_ROOT): checkpoint_volume,
        str(OUTPUT_ROOT): output_volume,
    },
)
def prepare_data(material_limit: int = 100, download_materials: bool = True) -> dict:
    """Populate Modal volumes with the default SceneSmith runtime assets."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    _download_sam3d_checkpoints()
    _download_artvip()
    if download_materials:
        _download_materials(material_limit=material_limit)
        _download_material_embeddings()

    data_volume.commit()
    checkpoint_volume.commit()
    output_volume.commit()

    return {
        "data_volume": DATA_VOLUME_NAME,
        "checkpoint_volume": CHECKPOINT_VOLUME_NAME,
        "output_volume": OUTPUT_VOLUME_NAME,
        "materials_limited_to": material_limit if material_limit > 0 else None,
    }


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=16,
    memory=65536,
    timeout=GENERATE_TIMEOUT_S,
    startup_timeout=GENERATE_STARTUP_TIMEOUT_S,
    secrets=[runtime_secret],
    volumes={
        str(DATA_ROOT): data_volume,
        str(CHECKPOINT_ROOT): checkpoint_volume,
        str(OUTPUT_ROOT): output_volume,
    },
)
def generate_scene(
    prompt: str,
    mode: str = "room",
    run_name: str | None = None,
    start_stage: str = "floor_plan",
    stop_stage: str = "manipuland",
    resume_from_path: str = "",
    resume_run_id: str = "",
    extra_overrides: str = "",
) -> dict:
    """Run SceneSmith on Modal for a single prompt."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if mode not in {"room", "house"}:
        raise ValueError("mode must be 'room' or 'house'")
    _validate_stage(start_stage, "start_stage")
    _validate_stage(stop_stage, "stop_stage")

    start_idx = PIPELINE_STAGES.index(start_stage)
    stop_idx = PIPELINE_STAGES.index(stop_stage)
    if start_idx > stop_idx:
        raise ValueError(
            f"start_stage {start_stage!r} cannot come after stop_stage {stop_stage!r}"
        )

    resume_source = _resolve_resume_source(resume_from_path, resume_run_id)
    if resume_source and start_stage == "floor_plan":
        raise ValueError("Resuming requires start_stage after 'floor_plan'")

    _ensure_runtime_prereqs()

    run_id = _sanitize_run_name(run_name)
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "prompts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_index", "prompt"])
        writer.writerow([0, prompt])

    cmd = [
        str(APP_PYTHON),
        "main.py",
        f"+name={run_id}",
        f"hydra.run.dir={run_dir}",
        f"experiment.csv_path={csv_path}",
        "experiment.num_workers=1",
        f"floor_plan_agent.mode={mode}",
        f"experiment.pipeline.start_stage={start_stage}",
        f"experiment.pipeline.stop_stage={stop_stage}",
    ]
    if resume_source:
        cmd.append(f"experiment.pipeline.resume_from_path={resume_source}")
    if extra_overrides.strip():
        cmd.extend(shlex.split(extra_overrides))

    _run(cmd, env=_runtime_env())
    output_volume.commit()

    return {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "output_volume": OUTPUT_VOLUME_NAME,
        "start_stage": start_stage,
        "stop_stage": stop_stage,
        "resume_from_path": resume_source,
    }


@app.local_entrypoint()
def main(
    prompt: str = "",
    mode: str = "room",
    prepare: bool = False,
    material_limit: int = 100,
    run_name: str = "",
    start_stage: str = "floor_plan",
    stop_stage: str = "manipuland",
    resume_from_path: str = "",
    resume_run_id: str = "",
    extra_overrides: str = "",
) -> None:
    """Optional convenience wrapper for local `modal run` usage."""
    if prepare:
        result = prepare_data.spawn(material_limit=material_limit).get()
        print(result)

    if prompt:
        result = generate_scene.spawn(
            prompt=prompt,
            mode=mode,
            run_name=run_name or None,
            start_stage=start_stage,
            stop_stage=stop_stage,
            resume_from_path=resume_from_path,
            resume_run_id=resume_run_id,
            extra_overrides=extra_overrides,
        ).get()
        print(result)
        return

    if not prepare:
        raise ValueError(
            "Provide --prompt to generate a scene, pass --prepare to only download "
            "runtime data, or call ::prepare_data directly."
        )
