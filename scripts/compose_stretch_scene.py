#!/usr/bin/env python3
"""Compose a SceneSmith MuJoCo scene with the Stretch robot model.

This creates a single self-contained MJCF file that can be passed to
`stretch_mujoco` via `--scene-xml-path`.

Example:
    uv run python scripts/compose_stretch_scene.py \
        --scene-xml scene_mujoco/scene.xml \
        --stretch-pos 2.0 -1.0 0.0 \
        --stretch-yaw-deg 90 \
        --output scene_mujoco/scene_with_stretch.xml
"""

from __future__ import annotations

import argparse
import copy
import math
import xml.etree.ElementTree as ET

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE_XML = REPO_ROOT / "scene_mujoco" / "scene.xml"
DEFAULT_STRETCH_XML = (
    REPO_ROOT.parent / "stretch_mujoco" / "stretch_mujoco" / "models" / "stretch.xml"
)

ASSET_TAGS = {"mesh", "texture", "hfield", "skin"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a SceneSmith MuJoCo scene with stretch_mujoco/stretch.xml"
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=DEFAULT_SCENE_XML,
        help=f"Path to SceneSmith scene.xml (default: {DEFAULT_SCENE_XML})",
    )
    parser.add_argument(
        "--stretch-xml",
        type=Path,
        default=DEFAULT_STRETCH_XML,
        help=f"Path to stretch.xml (default: {DEFAULT_STRETCH_XML})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=REPO_ROOT / "scene_mujoco" / "scene_with_stretch.xml",
        help="Output path for the merged MJCF",
    )
    parser.add_argument(
        "--stretch-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional Stretch base position to bake into the merged MJCF.",
    )
    parser.add_argument(
        "--stretch-yaw-deg",
        type=float,
        default=None,
        help="Optional Stretch base yaw in degrees. Ignored if --stretch-quat is set.",
    )
    parser.add_argument(
        "--stretch-quat",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="Optional Stretch base quaternion in MuJoCo scalar-first order.",
    )
    args = parser.parse_args()
    if args.stretch_yaw_deg is not None and args.stretch_quat is not None:
        parser.error("Use either --stretch-yaw-deg or --stretch-quat, not both.")
    return args


def _load_xml(path: Path) -> ET.Element:
    if not path.exists():
        raise FileNotFoundError(path)
    return ET.parse(path).getroot()


def _resolve_asset_file(
    elem: ET.Element,
    *,
    xml_dir: Path,
    compiler: ET.Element | None,
) -> None:
    if elem.tag == "texture":
        elem.attrib.pop("colorspace", None)

    file_attr = elem.get("file")
    if not file_attr:
        return

    file_path = Path(file_attr)
    if file_path.is_absolute():
        return

    compiler_attrs = compiler.attrib if compiler is not None else {}
    if elem.tag == "mesh":
        rel_dir = compiler_attrs.get("meshdir") or compiler_attrs.get("assetdir")
    elif elem.tag == "texture":
        rel_dir = compiler_attrs.get("texturedir") or compiler_attrs.get("assetdir")
    elif elem.tag in ASSET_TAGS:
        rel_dir = compiler_attrs.get("assetdir")
    else:
        rel_dir = None

    base_dir = xml_dir / rel_dir if rel_dir else xml_dir
    elem.set("file", str((base_dir / file_path).resolve()))


def _copy_children(parent: ET.Element | None) -> list[ET.Element]:
    if parent is None:
        return []
    return [copy.deepcopy(child) for child in list(parent)]


def _merged_compiler(
    scene_compiler: ET.Element | None,
    stretch_compiler: ET.Element | None,
) -> ET.Element | None:
    if scene_compiler is None and stretch_compiler is None:
        return None

    merged = ET.Element("compiler")
    for compiler in (scene_compiler, stretch_compiler):
        if compiler is None:
            continue
        for key, value in compiler.attrib.items():
            if key in {"meshdir", "texturedir", "assetdir"}:
                continue
            merged.set(key, value)
    return merged


def _normalize_assets(
    asset_root: ET.Element,
    *,
    xml_dir: Path,
    compiler: ET.Element | None,
) -> ET.Element:
    normalized = ET.Element("asset")
    for child in _copy_children(asset_root):
        _resolve_asset_file(child, xml_dir=xml_dir, compiler=compiler)
        normalized.append(child)
    return normalized


def _format_floats(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _yaw_quat_deg(yaw_deg: float) -> list[float]:
    half_yaw = math.radians(yaw_deg) / 2.0
    return [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]


def _apply_stretch_pose(
    stretch_worldbody: ET.Element | None,
    *,
    stretch_pos: list[float] | None,
    stretch_quat: list[float] | None,
) -> None:
    if stretch_worldbody is None:
        return

    base_link = stretch_worldbody.find("./body[@name='base_link']")
    if base_link is None:
        raise ValueError("Stretch worldbody does not contain base_link.")

    if stretch_pos is not None:
        base_link.set("pos", _format_floats(stretch_pos))

    if stretch_quat is not None:
        base_link.set("quat", _format_floats(stretch_quat))


def compose_scene(
    *,
    scene_xml_path: Path,
    stretch_xml_path: Path,
    output_path: Path,
    stretch_pos: list[float] | None = None,
    stretch_quat: list[float] | None = None,
) -> Path:
    scene_root = _load_xml(scene_xml_path)
    stretch_root = _load_xml(stretch_xml_path)

    scene_compiler = scene_root.find("compiler")
    stretch_compiler = stretch_root.find("compiler")

    merged_root = ET.Element(
        "mujoco",
        {"model": f"{scene_root.get('model', 'scene')}_with_stretch"},
    )

    merged_compiler = _merged_compiler(scene_compiler, stretch_compiler)
    if merged_compiler is not None:
        merged_root.append(merged_compiler)

    for tag in ("option", "size", "default"):
        stretch_elem = stretch_root.find(tag)
        if stretch_elem is not None:
            merged_root.append(copy.deepcopy(stretch_elem))

    scene_visual = scene_root.find("visual")
    stretch_visual = stretch_root.find("visual")
    if scene_visual is not None:
        merged_root.append(copy.deepcopy(scene_visual))
    elif stretch_visual is not None:
        merged_root.append(copy.deepcopy(stretch_visual))

    merged_asset = ET.Element("asset")
    scene_asset = scene_root.find("asset")
    if scene_asset is not None:
        for child in list(
            _normalize_assets(
                scene_asset,
                xml_dir=scene_xml_path.parent,
                compiler=scene_compiler,
            )
        ):
            merged_asset.append(child)

    stretch_asset = stretch_root.find("asset")
    if stretch_asset is not None:
        for child in list(
            _normalize_assets(
                stretch_asset,
                xml_dir=stretch_xml_path.parent,
                compiler=stretch_compiler,
            )
        ):
            merged_asset.append(child)

    if list(merged_asset):
        merged_root.append(merged_asset)

    merged_worldbody = ET.Element("worldbody")
    scene_worldbody = scene_root.find("worldbody")
    stretch_worldbody = stretch_root.find("worldbody")
    _apply_stretch_pose(
        stretch_worldbody,
        stretch_pos=stretch_pos,
        stretch_quat=stretch_quat,
    )
    for worldbody in (scene_worldbody, stretch_worldbody):
        for child in _copy_children(worldbody):
            merged_worldbody.append(child)
    merged_root.append(merged_worldbody)

    for tag in ("contact", "tendon", "equality", "actuator", "sensor", "keyframe"):
        stretch_elem = stretch_root.find(tag)
        if stretch_elem is not None:
            merged_root.append(copy.deepcopy(stretch_elem))

    ET.indent(merged_root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged_root).write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def main() -> None:
    args = _parse_args()
    stretch_quat = None
    if args.stretch_quat is not None:
        stretch_quat = list(args.stretch_quat)
    elif args.stretch_yaw_deg is not None:
        stretch_quat = _yaw_quat_deg(args.stretch_yaw_deg)

    output_path = compose_scene(
        scene_xml_path=args.scene_xml.resolve(),
        stretch_xml_path=args.stretch_xml.resolve(),
        output_path=args.output.resolve(),
        stretch_pos=list(args.stretch_pos) if args.stretch_pos is not None else None,
        stretch_quat=stretch_quat,
    )
    print(output_path)


if __name__ == "__main__":
    main()
