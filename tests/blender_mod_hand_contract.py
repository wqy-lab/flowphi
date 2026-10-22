"""Run inside Blender with the converted MOD-FlowPhi.blend open."""
import bpy
from mathutils import Vector
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowphi.rendering.blender import configure_camera, bake_object_transforms

# Bulk-written object curves must evaluate like the old per-sample API.
old, bulk = [bpy.data.objects.new(name, None) for name in ('Old bake test', 'Bulk bake test')]
for obj in (old, bulk):
    bpy.context.scene.collection.objects.link(obj)
    obj.rotation_mode = 'QUATERNION'
samples = {'location': [], 'rotation_quaternion': []}
for frame, position in [(1., (0., 0., 0.)), (2.2, (.1, -.2, .3)), (4., (.2, .3, .1))]:
    old.location = position
    old.rotation_quaternion = (1., 0., 0., 0.)
    for path in samples:
        old.keyframe_insert(path, frame=frame)
        samples[path].append((frame, tuple(getattr(old, path))))
for curve in old.animation_data.action.fcurves:
    for key in curve.keyframe_points:
        key.interpolation = 'LINEAR'
bake_object_transforms((bulk,), {bulk.name: samples})
for frame in range(1,5):
    bpy.context.scene.frame_set(frame)
    assert max(abs(a-b) for ar,br in zip(old.matrix_world,bulk.matrix_world) for a,b in zip(ar,br)) < 1e-6

assert bpy.context.scene.camera is None
job = dict(camera_projection='perspective', view_width=.35, camera_y=-.015)
configure_camera(bpy.context.scene, job)
camera = bpy.context.scene.camera
assert camera is not None and camera.data.type == 'PERSP'
configure_camera(bpy.context.scene, dict(job, camera_projection='orthographic'))
assert bpy.context.scene.camera == camera and camera.data.type == 'ORTHO'

chains = [('Bone' if i == 0 else f'Bone.{i:03d}') for i in range(12)]
chains += [f'Bone.{i:03d}' for i in (17, 18, 19)]
left, right = [bpy.data.objects[f'Hand_{side}'] for side in ('Left', 'Right')]
meshes = [next(o for o in bpy.data.objects if o.type == 'MESH' and o.parent == arm)
          for arm in (left, right)]
assert len(meshes[0].data.vertices) == len(meshes[1].data.vertices)
for a, b in zip(meshes[0].data.vertices, meshes[1].data.vertices):
    assert (Vector((-a.co.x, a.co.y, a.co.z)) - b.co).length < 1e-6
for arm, mesh in zip((left, right), meshes):
    assert set(chains + [f'Bone.{i:03d}' for i in (12, 13, 14, 15, 16, 20, 21)]) <= set(arm.data.bones.keys())
    for name in chains:
        group = mesh.vertex_groups[name].index
        assert any(any(g.group == group and g.weight > .5 for g in v.groups) for v in mesh.data.vertices)
        before = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        positions = [v.co.copy() for v in before.vertices]
        mesh.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
        bone = arm.pose.bones[name]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler.x = -.2
        bpy.context.view_layer.update()
        evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        after = evaluated.to_mesh()
        assert max((a - b.co).length for a, b in zip(positions, after.vertices)) > 1e-4, name
        evaluated.to_mesh_clear()
        bone.rotation_euler.x = 0
        bpy.context.view_layer.update()
print('MOD_HAND_CONTRACT_OK: mirrored meshes and 30 deforming finger joints')
