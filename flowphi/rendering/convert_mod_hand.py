"""Convert MOD's HandRigs.blend human hand to the FlowPhi rig contract.

Run in Blender with the original file open:
  blender -b HandRigs.blend --disable-autoexec --python convert_mod_hand.py -- output.blend
The source file remains unchanged. No Rigify add-on or embedded scripts are run.
"""
from pathlib import Path
import math
import sys


def convert(destination):
    import bpy
    from mathutils import Matrix, Vector

    source_arm = bpy.data.objects.get('Hand Rig')
    source_mesh = bpy.data.objects.get('Hand')
    if source_arm is None or source_mesh is None or source_arm.type != 'ARMATURE':
        raise ValueError('Expected MOD HandRigs.blend with Hand and Hand Rig')
    mapping = {'hand.L': 'Bone.016'}
    for stem, first in [('f_index', 0), ('f_middle', 3), ('f_ring', 6),
                        ('f_pinky', 9), ('thumb', 17)]:
        for joint in range(3):
            number = first + joint
            mapping[f'{stem}.{joint + 1:02d}.L'] = 'Bone' if number == 0 else f'Bone.{number:03d}'
    mapping.update({f'palm.{i + 1:02d}.L': f'Bone.{12 + i:03d}' for i in range(4)})
    missing = set(mapping) - set(source_arm.data.bones.keys())
    if missing or set(mapping) - set(source_mesh.vertex_groups.keys()):
        raise ValueError(f'Missing expected human-hand bones or skin weights: {missing}')
    # Source fingers point +Z; FlowPhi expects -Y and the back of the hand +Z.
    wrist = source_arm.data.bones['hand.L'].head_local.copy()
    transform = Matrix.Rotation(math.pi / 2, 4, 'X') @ Matrix.Translation(-wrist)
    bones = {name: (transform @ b.head_local, transform @ b.tail_local,
                   b.parent.name if b.parent else None)
             for name, b in source_arm.data.bones.items() if name in mapping}
    mesh_data = source_mesh.data.copy()
    mesh_data.transform(transform @ source_mesh.matrix_world)
    groups = [(g.name, g.index) for g in source_mesh.vertex_groups]
    weights = [[(g.group, g.weight) for g in v.groups] for v in source_mesh.data.vertices]
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for side, suffix in [(-1, 'Left'), (1, 'Right')]:
        mirror = Matrix.Diagonal((1 if side == -1 else -1, 1, 1, 1))
        arm_data = bpy.data.armatures.new(f'MOD_{suffix}')
        arm = bpy.data.objects.new(f'Hand_{suffix}', arm_data)
        bpy.context.scene.collection.objects.link(arm)
        bpy.context.view_layer.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        for name, (head, tail, _) in bones.items():
            b = arm_data.edit_bones.new(mapping[name])
            b.head, b.tail = mirror @ head, mirror @ tail
            b.align_roll(Vector((0, 0, 1)))
        for name, (_, _, parent) in bones.items():
            if parent in mapping:
                arm_data.edit_bones[mapping[name]].parent = arm_data.edit_bones[mapping[parent]]
        # Legacy renderer models thumb metacarpal and forearm separately.
        palm = arm_data.edit_bones['Bone.016']
        thumb = arm_data.edit_bones['Bone.017']
        base = arm_data.edit_bones.new('Bone.021')
        base.head, base.tail = palm.head, thumb.head
        base.parent = palm
        base.align_roll(Vector((0, 0, 1)))
        thumb.parent = base
        forearm = arm_data.edit_bones.new('Bone.020')
        forearm.head, forearm.tail = palm.head + Vector((0, .12, 0)), palm.head
        forearm.align_roll(Vector((0, 0, 1)))
        palm.parent = forearm
        bpy.ops.object.mode_set(mode='OBJECT')
        arm.select_set(False)
        data = mesh_data.copy()
        data.transform(mirror)
        if side == 1:
            for polygon in data.polygons:
                polygon.flip()
        mesh = bpy.data.objects.new(f'MOD_Hand_{suffix}', data)
        bpy.context.scene.collection.objects.link(mesh)
        # Mesh copies retain group indices; recreate them in their original order.
        mesh.vertex_groups.clear()
        for name, _ in groups:
            mesh.vertex_groups.new(name=mapping.get(name, name))
        for vertex, entries in enumerate(weights):
            for group, weight in entries:
                mesh.vertex_groups[group].add([vertex], weight, 'REPLACE')
        for name in ('Bone.020', 'Bone.021'):
            mesh.vertex_groups.new(name=name)
        mesh.parent = arm
        modifier = mesh.modifiers.new('FlowPhi skin', 'ARMATURE')
        modifier.object = arm
        modifier.use_deform_preserve_volume = True
        smooth = mesh.modifiers.new('Surface smoothing', 'SUBSURF')
        smooth.levels = smooth.render_levels = 1
        arm['flowphi_adapter'] = 'MOD HandRigs v1'
        # Fail before saving if a finger has no usable weighted vertices.
        for name in (mapping[f'{stem}.{joint:02d}.L'] for stem in
                     ('f_index', 'f_middle', 'f_ring', 'f_pinky', 'thumb') for joint in (1, 2, 3)):
            group = mesh.vertex_groups[name].index
            if not any(any(g.group == group and g.weight > .5 for g in v.groups)
                       for v in mesh.data.vertices):
                raise ValueError(f'No dominant skin weights for {suffix}/{name}')
    bpy.context.scene['model_source'] = 'https://mightbemod.itch.io/simple-blender-hand-models'
    bpy.context.scene['model_author'] = 'MOD'
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination == Path(bpy.data.filepath).resolve():
        raise ValueError('Choose a separate output file; preserve the source model')
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    print('FLOWPHI_MOD_ADAPTER_OK', destination)


if __name__ == '__main__':
    convert(sys.argv[sys.argv.index('--') + 1])
