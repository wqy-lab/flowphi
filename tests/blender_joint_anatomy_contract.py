"""Run in Blender with MOD-FlowPhi.blend, including subframe playback."""
import bpy
import math
from mathutils import Euler
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowphi import hand_limits as limits
from flowphi.rendering.blender import FINGERS, bake_rotations

rigs = {side: bpy.data.objects[name] for side, name in ((-1, 'Hand_Left'), (1, 'Hand_Right'))}
baked, models, references = {}, {}, {}
for side, arm in rigs.items():
    arm.animation_data_clear()
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions.new(f'Anatomy contract {side}')
    for finger, names in FINGERS.items():
        for joint, name in enumerate(names):
            bone = arm.pose.bones[name]
            for constraint in list(bone.constraints):
                bone.constraints.remove(constraint)
            reference = tuple(Euler((.12 if joint == 0 else -.4, 0., -side*.08 if joint == 0 else 0.), 'XYZ').to_quaternion())
            model = limits.joint_model(finger, joint,
                spread_axis=limits.apply(limits.conjugate(reference), (0., 0., 1.)))
            models[side, name], references[side, name] = model, reference
            previous, last_frame = reference, 1.
            keys = [(1., reference)]
            for frame, sign in ((4.25, 1), (9.9, -1)):
                # Deliberately requests both sideways motion and forbidden roll.
                desired = limits.multiply(reference, limits.multiply(
                    limits.axis_rotation(model.spread, sign*1.3), limits.axis_rotation(model.forward, .9)))
                value = limits.joint_rotation(previous, desired, reference, joint, (frame-last_frame)/30, model)
                keys.append((frame, value))
                previous, last_frame = value, frame
            baked[side, name] = keys
bake_rotations(rigs, baked)
maximum = 0.
last = {}
for index in range(891):
    frame = 1+index/100
    bpy.context.scene.frame_set(int(frame), subframe=frame-int(frame))
    for key, model in models.items():
        q = tuple(rigs[key[0]].pose.bones[key[1]].rotation_euler.to_quaternion())
        reference = references[key]
        residual = limits.angular_distance(q, limits.project_joint(q, reference, model))
        maximum = max(maximum, residual)
        assert residual < 3e-6, (frame, key, residual, q, reference,
                               tuple(rigs[key[0]].pose.bones[key[1]].rotation_euler))
        if key in last:
            speed = limits.angular_distance(last[key], q)/(.01/30)
            limit = limits.MCP_SPEED if model.lateral else limits.DISTAL_SPEED
            assert speed < limit+.01, (frame, key, speed)
        last[key] = q
print('ANATOMY_BAKE_CONTRACT_OK', 'bones', len(models), 'max_subframe_residual_rad', maximum, flush=True)
