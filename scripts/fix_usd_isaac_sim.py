"""Fix USD physics for Isaac Sim compatibility.

The mujoco-usd-converter (v0.1.0a3) generates PhysicsFixedJoint prims that
connect objects to the root Xform, but the root has no PhysicsRigidBodyAPI.
PhysX requires valid physics bodies on both sides of a joint, so the
constraint solver pulls everything to (0,0,0).

This script post-processes Physics.usda files to fix three object categories:

1. **Static objects** (walls, desks, beds): Remove all physics body APIs and
   joints, leaving only collision geometry. Isaac Sim treats these as static
   colliders.

2. **Dynamic objects** (mugs, books): Flatten nested rigid bodies by moving
   MassAPI from base_link to wrapper, removing inner RigidBodyAPI, and
   deleting the internal FixedJoint.

3. **Articulated objects** (wardrobes with doors, fridges): Reparent nested
   rigid bodies as siblings, add self-collision filters (mirroring MuJoCo's
   ``<contact><exclude>`` pairs), and set up correct articulation structure.

Usage:
    # Fix single scene USD directory.
    python scripts/fix_usd_isaac_sim.py /path/to/scene/mujoco/usd

    # Fix all scenes recursively with parallel workers.
    python scripts/fix_usd_isaac_sim.py /path/to/SceneAgent_Cleaned \\
        --recursive --workers 16
"""

import argparse
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics

console_logger = logging.getLogger(__name__)


def remove_rigid_body_api(prim: Usd.Prim) -> bool:
    """Remove PhysicsRigidBodyAPI from a prim if present."""
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        return True
    return False


def remove_mass_api(prim: Usd.Prim) -> None:
    """Remove PhysicsMassAPI and all mass properties from a prim."""
    if not prim.HasAPI(UsdPhysics.MassAPI):
        return
    prim.RemoveAPI(UsdPhysics.MassAPI)
    for prop_name in [
        "physics:mass",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    ]:
        prop = prim.GetProperty(prop_name)
        if prop:
            prim.RemoveProperty(prop_name)


def copy_mass_to_prim(source: Usd.Prim, target: Usd.Prim) -> None:
    """Copy PhysicsMassAPI and its properties from source to target prim."""
    if not source.HasAPI(UsdPhysics.MassAPI):
        return
    UsdPhysics.MassAPI.Apply(target)

    mass_props = [
        ("physics:mass", "float"),
        ("physics:centerOfMass", "point3f"),
        ("physics:diagonalInertia", "float3"),
        ("physics:principalAxes", "quatf"),
    ]
    for prop_name, _ in mass_props:
        src_attr = source.GetAttribute(prop_name)
        if src_attr and src_attr.HasValue():
            tgt_attr = target.GetAttribute(prop_name)
            if not tgt_attr:
                # Create with same type as source.
                tgt_attr = target.CreateAttribute(prop_name, src_attr.GetTypeName())
            tgt_attr.Set(src_attr.Get())


def find_fixed_joints_with_body0(
    root_prim: Usd.Prim, body0_path: Sdf.Path
) -> list[Sdf.Path]:
    """Find all PhysicsFixedJoint descendants whose body0 targets body0_path."""
    joint_paths = []
    for descendant in Usd.PrimRange(root_prim):
        if descendant.GetTypeName() == "PhysicsFixedJoint":
            body0_rel = descendant.GetRelationship("physics:body0")
            if body0_rel:
                targets = body0_rel.GetTargets()
                if targets and targets[0] == body0_path:
                    joint_paths.append(descendant.GetPath())
    return joint_paths


def delete_prims(stage: Usd.Stage, paths: list[Sdf.Path]) -> int:
    """Delete prims at the given paths. Returns count of deleted prims."""
    count = 0
    for path in paths:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
            count += 1
    return count


def _reparent_nested_rigid_bodies(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    all_layers: list[Sdf.Layer],
) -> None:
    """Move nested rigid bodies to be direct children of wrapper.

    The MuJoCo→USD converter creates a hierarchy like:
        wrapper/E_body/E_door  (door is child of body)
    But PhysX requires articulation links to be siblings:
        wrapper/E_body
        wrapper/E_door

    We reparent across ALL sublayers (Physics, Geometry, Materials) so
    that mesh data, materials, and physics all move together. Joint
    relationship targets are updated in the stage's edit target layer.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Find the root body (first direct child with RigidBodyAPI).
    root_body = None
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            root_body = child
            break
    if root_body is None:
        return

    # Collect child rigid bodies that need reparenting.
    prims_to_move = []
    for child in root_body.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            prims_to_move.append(child.GetPath())

    if not prims_to_move:
        return

    # Build path mapping.
    path_mapping: dict[str, str] = {}
    for old_path in prims_to_move:
        new_path = wrapper_path.AppendChild(old_path.name)
        path_mapping[str(old_path)] = str(new_path)

    # Apply reparenting to EVERY layer that contains these prims.
    for layer in all_layers:
        edit = Sdf.BatchNamespaceEdit()
        has_edits = False
        for old_path in prims_to_move:
            if layer.GetPrimAtPath(old_path):
                new_path = wrapper_path.AppendChild(old_path.name)
                edit.Add(old_path, new_path)
                has_edits = True
        if has_edits:
            if not layer.Apply(edit):
                console_logger.warning(
                    f"Failed to reparent in layer {layer.identifier}"
                )

    # Update all relationship targets that referenced the old paths.
    # This covers joint body0/body1 targets.
    for descendant in Usd.PrimRange(wrapper_prim):
        for rel in descendant.GetRelationships():
            targets = rel.GetTargets()
            new_targets = []
            changed = False
            for target in targets:
                target_str = str(target)
                for old_str, new_str in path_mapping.items():
                    if target_str == old_str or target_str.startswith(old_str + "/"):
                        target_str = new_str + target_str[len(old_str) :]
                        changed = True
                        break
                new_targets.append(Sdf.Path(target_str))
            if changed:
                rel.SetTargets(new_targets)


def _add_self_collision_filter(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> None:
    """Add self-collision filtering for all rigid bodies in an articulated object.

    The MuJoCo source has ``<contact><exclude>`` pairs that prevent adjacent
    articulated links from colliding (e.g. wardrobe body vs. its doors).
    The mujoco_usd_converter does not convert these (``Tf.Warn("excludes
    are not supported")``), so we recreate them using a PhysicsCollisionGroup
    that includes all rigid bodies within the object and filters against
    itself.

    Without this, PhysX detects collisions between overlapping bodies at
    hinge points, which prevents joints from moving interactively.
    """
    # Collect all rigid body prims under the wrapper.
    rigid_bodies = []
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(descendant.GetPath())

    if len(rigid_bodies) < 2:
        return  # No self-collision possible with fewer than 2 bodies.

    # Create a PhysicsCollisionGroup under the wrapper.
    group_path = wrapper_prim.GetPath().AppendChild("selfCollisionFilter")
    group = UsdPhysics.CollisionGroup.Define(stage, group_path)

    # Add all rigid bodies to the group via CollectionAPI.
    collection = group.GetCollidersCollectionAPI()
    includes_rel = collection.CreateIncludesRel()
    for body_path in rigid_bodies:
        includes_rel.AddTarget(body_path)

    # Filter the group against itself → disables collision between members.
    filtered_rel = group.GetFilteredGroupsRel()
    filtered_rel.AddTarget(group_path)

    console_logger.debug(
        f"  {wrapper_prim.GetPath().name}: self-collision filter for "
        f"{len(rigid_bodies)} bodies"
    )


def _has_nested_rigid_bodies(wrapper_prim: Usd.Prim) -> bool:
    """Check if any child rigid body has a child that is also a rigid body."""
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            for grandchild in child.GetChildren():
                if grandchild.HasAPI(UsdPhysics.RigidBodyAPI):
                    return True
    return False


def classify_object(
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> str:
    """Classify an object as 'static', 'dynamic', or 'articulated'.

    Classification logic:
    1. Check ArticulationRootAPI first — articulated objects may not have
       FixedJoints to root (e.g. when furniture uses freejoints in MuJoCo).
    2. Check for nested rigid bodies — this catches partially-fixed objects
       from prior runs where ArticulationRootAPI was already removed but
       bodies were not yet reparented as siblings.
    3. Check if wrapper has a FixedJoint descendant with body0 targeting root.
    4. If welded and no ArticulationRootAPI -> 'static'.
    5. If not welded and no ArticulationRootAPI -> 'dynamic'.
    """
    if wrapper_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return "articulated"
    if _has_nested_rigid_bodies(wrapper_prim):
        return "articulated"
    welded_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    if welded_joints:
        return "static"
    return "dynamic"


def fix_static_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> None:
    """Fix a static object by removing all physics body APIs and joints.

    Leaves only PhysicsCollisionAPI on collision geometry, making the object
    a static collider in Isaac Sim.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Remove RigidBodyAPI from wrapper.
    remove_rigid_body_api(wrapper_prim)

    # Remove RigidBodyAPI + MassAPI from all descendants.
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetPath() == wrapper_path:
            continue
        remove_rigid_body_api(descendant)
        remove_mass_api(descendant)

    # Delete FixedJoint from wrapper to root.
    root_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    delete_prims(stage, root_joints)

    # Delete FixedJoint from base_link/body_link to wrapper.
    inner_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, inner_joints)


def fix_dynamic_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> None:
    """Fix a dynamic object by flattening to a single rigid body.

    Moves MassAPI from base_link to wrapper and removes the inner
    RigidBodyAPI and FixedJoint.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Find the immediate child (base_link) that has MassAPI.
    base_link = None
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.MassAPI):
            base_link = child
            break

    if base_link is None:
        console_logger.warning(
            f"Dynamic object {wrapper_path} has no child with MassAPI, "
            "skipping mass copy."
        )
    else:
        # Copy mass properties from base_link to wrapper.
        copy_mass_to_prim(source=base_link, target=wrapper_prim)
        # Remove MassAPI + RigidBodyAPI from base_link.
        remove_mass_api(base_link)
        remove_rigid_body_api(base_link)

    # Delete FixedJoint inside base_link (base_link→wrapper).
    inner_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, inner_joints)


def fix_articulated_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
    all_layers: list[Sdf.Layer],
) -> None:
    """Fix an articulated object for Isaac Sim / PhysX compatibility.

    The converter creates two types of articulated objects:

    A) **Freejoint** (wardrobes, cabinets): No FixedJoint to scene root.
       These should be free-floating — the whole object can be pushed
       around. Fix: remove ArticulationRootAPI entirely and delete the
       internal FixedJoint. Bodies become regular rigid bodies connected
       by joints. This allows Isaac Sim's interactive force tools to
       work on each body individually.

    B) **Welded** (wall-mounted sconces, built-in cabinets): Has a
       FixedJoint to scene root. These should be fixed-base articulations.
       Fix: clear body0 on the FixedJoint from wrapper→E_body so it
       anchors E_body to the world.

    Common fixes for both:
    - Remove RigidBodyAPI from the wrapper.
    - Delete any FixedJoint from wrapper to the scene root (invalid
      because the root Xform has no RigidBodyAPI).
    - Move child rigid bodies (doors/drawers) from being nested under
      E_body to being direct children of the wrapper (PhysX requires
      sibling rigid bodies, not parent-child nesting).
    """
    wrapper_path = wrapper_prim.GetPath()

    # Determine if this object is welded to world or free-floating.
    # Welded objects have a FixedJoint from wrapper to the scene root.
    root_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    is_welded = len(root_joints) > 0

    # 1. Remove RigidBodyAPI from the wrapper.
    remove_rigid_body_api(wrapper_prim)

    # 2. Find the FixedJoint from wrapper→E_body.
    wrapper_to_body_joints: list[Sdf.Path] = []
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetTypeName() == "PhysicsFixedJoint":
            body0_rel = descendant.GetRelationship("physics:body0")
            if body0_rel:
                targets = body0_rel.GetTargets()
                if targets and targets[0] == wrapper_path:
                    wrapper_to_body_joints.append(descendant.GetPath())

    if is_welded:
        # Fixed-base articulation: keep ArticulationRootAPI, clear body0
        # to world-anchor E_body.
        for jp in wrapper_to_body_joints:
            joint_prim = stage.GetPrimAtPath(jp)
            if joint_prim:
                body0_rel = joint_prim.GetRelationship("physics:body0")
                if body0_rel:
                    body0_rel.ClearTargets(True)
        console_logger.debug(f"  {wrapper_path.name}: fixed-base (welded to world)")
    else:
        # Free-floating: remove ArticulationRootAPI so bodies are regular
        # rigid bodies connected by joints. This allows Isaac Sim's
        # interactive force tools to work on each body. Delete the
        # internal FixedJoint (not needed without an articulation).
        if wrapper_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            wrapper_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        delete_prims(stage, wrapper_to_body_joints)
        console_logger.debug(f"  {wrapper_path.name}: free-floating (no articulation)")

    # 3. Delete FixedJoint from wrapper to scene root (invalid target).
    delete_prims(stage, root_joints)

    # 4. Reparent child rigid bodies from E_body to wrapper so they are
    #    siblings across ALL layers (Physics, Geometry, Materials).
    _reparent_nested_rigid_bodies(stage, wrapper_prim, all_layers)

    # 5. Add self-collision filtering between all rigid bodies within this
    #    object. Mirrors MuJoCo's <contact><exclude> pairs that prevent
    #    adjacent links from colliding (e.g. wardrobe body vs. door).
    _add_self_collision_filter(stage, wrapper_prim)


def fix_physics_layer(physics_usda_path: Path) -> dict[str, int]:
    """Fix physics in a Physics.usda file for Isaac Sim compatibility.

    Opens the composed stage and fixes all objects. For articulated
    objects, reparenting is applied across ALL sublayers (Physics,
    Geometry, Materials) so mesh data and materials move with the prims.

    Args:
        physics_usda_path: Path to the Physics.usda file.

    Returns:
        Dict with counts of objects fixed per category.
    """
    stage = Usd.Stage.Open(str(physics_usda_path))
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise RuntimeError(f"No default prim in {physics_usda_path}")

    root_path = root_prim.GetPath()

    # Find the Geometry scope.
    geometry_path = root_path.AppendChild("Geometry")
    geometry_prim = stage.GetPrimAtPath(geometry_path)
    if not geometry_prim:
        raise RuntimeError(f"No Geometry scope found at {geometry_path}")

    # Collect ALL sublayers in the Payload directory for reparenting.
    # The Payload dir contains Physics.usda, Geometry.usda, Materials.usda.
    payload_dir = physics_usda_path.parent
    all_layers: list[Sdf.Layer] = []
    for usda_file in sorted(payload_dir.glob("*.usda")):
        layer = Sdf.Layer.FindOrOpen(str(usda_file))
        if layer:
            all_layers.append(layer)

    counts: dict[str, int] = {"static": 0, "dynamic": 0, "articulated": 0}

    for wrapper_prim in geometry_prim.GetChildren():
        category = classify_object(
            wrapper_prim=wrapper_prim,
            root_path=root_path,
        )
        counts[category] += 1

        if category == "static":
            fix_static_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
                root_path=root_path,
            )
        elif category == "dynamic":
            fix_dynamic_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
            )
        elif category == "articulated":
            fix_articulated_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
                root_path=root_path,
                all_layers=all_layers,
            )

    # Save ALL modified layers (Physics + Geometry + Materials).
    stage.GetRootLayer().Save()
    for layer in all_layers:
        if layer.dirty:
            layer.Save()

    console_logger.info(
        f"Fixed {physics_usda_path}: "
        f"{counts['static']} static, "
        f"{counts['dynamic']} dynamic, "
        f"{counts['articulated']} articulated"
    )
    return counts


def _fix_single_scene(usd_dir: Path) -> tuple[Path, dict[str, int] | str]:
    """Fix a single scene's Physics.usda. Returns (path, counts_or_error)."""
    physics_path = usd_dir / "Payload" / "Physics.usda"
    if not physics_path.exists():
        return usd_dir, "no Physics.usda found"
    try:
        counts = fix_physics_layer(physics_path)
        return usd_dir, counts
    except Exception as e:
        return usd_dir, f"error: {e}"


def find_usd_dirs(base_path: Path, recursive: bool) -> list[Path]:
    """Find USD directories (containing Payload/Physics.usda)."""
    if not recursive:
        # Single scene: base_path should be the usd directory itself.
        if (base_path / "Payload" / "Physics.usda").exists():
            return [base_path]
        return []

    # Recursive: find all Physics.usda files.
    usd_dirs = []
    for physics_file in base_path.rglob("Payload/Physics.usda"):
        usd_dirs.append(physics_file.parent.parent)
    return sorted(usd_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix USD physics for Isaac Sim compatibility"
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a single USD directory (containing Payload/), "
            "or a parent directory when using --recursive"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for all USD scenes under the given path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for recursive mode (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    usd_dirs = find_usd_dirs(base_path=args.path, recursive=args.recursive)
    if not usd_dirs:
        console_logger.error(f"No USD scenes found at {args.path}")
        return

    console_logger.info(f"Found {len(usd_dirs)} USD scene(s) to fix")

    total_counts: dict[str, int] = {
        "static": 0,
        "dynamic": 0,
        "articulated": 0,
    }
    errors = 0

    if args.workers > 1 and len(usd_dirs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_fix_single_scene, d): d for d in usd_dirs}
            for future in as_completed(futures):
                path, result = future.result()
                if isinstance(result, str):
                    console_logger.warning(f"{path}: {result}")
                    errors += 1
                else:
                    for k, v in result.items():
                        total_counts[k] += v
    else:
        for usd_dir in usd_dirs:
            path, result = _fix_single_scene(usd_dir)
            if isinstance(result, str):
                console_logger.warning(f"{path}: {result}")
                errors += 1
            else:
                for k, v in result.items():
                    total_counts[k] += v

    console_logger.info(
        f"Done. Fixed {len(usd_dirs) - errors}/{len(usd_dirs)} scenes: "
        f"{total_counts['static']} static, "
        f"{total_counts['dynamic']} dynamic, "
        f"{total_counts['articulated']} articulated objects total"
    )
    if errors:
        console_logger.warning(f"{errors} scene(s) had errors")


if __name__ == "__main__":
    main()
