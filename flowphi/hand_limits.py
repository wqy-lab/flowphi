"""Shared output constraints for the skeleton preview and Blender rig.

Quaternions use (w, x, y, z). Limits are applied *after* contact/collision
fitting: an unreachable contact must remain an error, not override anatomy.
"""
import math
from dataclasses import dataclass

IDENTITY = (1., 0., 0., 0.)
WRIST_SPEED = .9                 # metres / second, full 3D displacement
WRIST_TURN_SPEED = 4.           # radians / second
MCP_SPEED = 8.
DISTAL_SPEED = 1.5
DISTAL_EXCURSION = .20          # radians from the relaxed middle/tip pose
ANATOMY_MODEL = 'hinge_and_bounded_mcp_v1'
# Conservative animation envelopes relative to the relaxed pose, not clinical ROM.
MCP_SPREAD = dict(index=math.radians(10), middle=math.radians(7),
                  ring=math.radians(8), little=math.radians(12), thumb=math.radians(35))


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _unit(v):
    length = math.hypot(*v)
    if length < 1e-12:
        raise ValueError('Degenerate anatomical joint axis')
    return tuple(x/length for x in v)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


@dataclass(frozen=True)
class JointModel:
    flex: tuple
    spread: tuple
    forward: tuple
    minimum: float
    maximum: float
    lateral: float


def joint_model(finger='index', joint=0, *, flex_axis=(1., 0., 0.), spread_axis=(0., 0., 1.)):
    """Axes live in the relaxed joint's coordinates; no longitudinal twist DOF."""
    flex = _unit(flex_axis)
    spread = _unit(tuple(x-_dot(spread_axis, flex)*y for x, y in zip(spread_axis, flex)))
    lower, upper = ((-1.1, .8) if finger == 'thumb' else (-1.2, .6)) if joint == 0 else (
        -DISTAL_EXCURSION, DISTAL_EXCURSION)
    return JointModel(flex, spread, _cross(spread, flex), lower, upper, MCP_SPREAD[finger] if joint == 0 else 0.)


def chain_models(rest, finger):
    """Derive flexion from each finger's neutral plane, including mirrored hands."""
    forward = tuple(b-a for a, b in zip(rest[0], rest[-1]))
    flat = (forward[0], forward[1], 0.)
    if math.hypot(*flat) < 1e-12:
        flat = (0., 1., 0.)
    flex = _unit(_cross(flat, (0., 0., 1.)))
    segment = _unit(tuple(b-a for a, b in zip(rest[0], rest[1])))
    spread = _unit(_cross(flex, segment))
    return tuple(joint_model(finger, i, flex_axis=flex, spread_axis=spread) for i in range(3))


def axis_rotation(axis, angle):
    return (math.cos(angle/2), *(x*math.sin(angle/2) for x in axis))


def joint_angles(q, reference, model):
    """Project arbitrary IK onto a hinge or a two-axis root, discarding twist."""
    delta = normalized(multiply(conjugate(reference), q))
    if model.lateral:
        direction = apply(delta, model.forward)
        lateral, forward = _dot(direction, model.flex), _dot(direction, model.forward)
        flex = math.atan2(_dot(direction, model.spread), math.hypot(lateral, forward))
        spread = math.atan2(-lateral, forward)
    else:
        # Closest rotation about the hinge, in the parent's neutral coordinates.
        flex = 2*math.atan2(_dot(delta[1:], model.flex), delta[0])
        flex = (flex+math.pi) % (2*math.pi)-math.pi
        spread = 0.
    return flex, spread


def joint_pose(reference, model, angles):
    flex, spread = angles
    return normalized(multiply(reference, multiply(axis_rotation(model.spread, spread), axis_rotation(model.flex, flex))))


def project_joint(q, reference, model):
    flex, spread = joint_angles(q, reference, model)
    return joint_pose(reference, model, (max(model.minimum, min(model.maximum, flex)),
                                       max(-model.lateral, min(model.lateral, spread))))


def joint_interpolate(a, b, u, reference, model):
    # A quaternion chord/slerp can leave the root's two-axis surface. Interpolate
    # its bounded coordinates instead; a hinge stays exactly on its single axis.
    if u <= 0 or a == b:
        return a
    if u >= 1:
        return b
    first, second = joint_angles(a, reference, model), joint_angles(b, reference, model)
    return joint_pose(reference, model, tuple(x+(y-x)*u for x, y in zip(first, second)))


def bounded_position(previous, desired, speed, dt):
    distance = math.dist(previous, desired)
    u = min(1., max(0., speed) * max(0., dt) / max(distance, 1e-15))
    return tuple(a + (b - a) * u for a, b in zip(previous, desired))


def normalized(q):
    w, x, y, z = q
    length = math.sqrt(w*w+x*x+y*y+z*z)
    return (w/length, x/length, y/length, z/length) if length > 1e-12 else IDENTITY


def conjugate(q):
    return q[0], -q[1], -q[2], -q[3]


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def angular_distance(a, b):
    # acos(dot) loses precision on the very short intervals between note events.
    w, x, y, z = multiply(conjugate(normalized(a)), normalized(b))
    sine = math.hypot(x, y, z)
    return 0. if sine < 1e-15 else 2 * math.atan2(sine, abs(w))


def interpolate(a, b, u):
    a, b = normalized(a), normalized(b)
    dot = a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
    if dot < 0:
        b, dot = tuple(-x for x in b), -dot
    if dot > .999999:
        return normalized(tuple(x + (y-x)*u for x, y in zip(a, b)))
    angle = math.acos(min(1., dot))
    return tuple((x*math.sin((1-u)*angle) + y*math.sin(u*angle)) / math.sin(angle) for x, y in zip(a, b))


def bounded_rotation(previous, desired, speed, dt, reference=None, excursion=None):
    desired = normalized(desired)
    if reference is not None and excursion is not None:
        angle = angular_distance(reference, desired)
        if angle > excursion:
            desired = interpolate(reference, desired, excursion / angle)
    if sum(a*b for a, b in zip(previous, desired)) < 0:
        desired = tuple(-x for x in desired)
    angle = angular_distance(previous, desired)
    maximum = max(0., speed)*max(0., dt)
    return desired if angle <= maximum else interpolate(previous, desired, maximum / angle)


def joint_rotation(previous, desired, reference, joint, dt, model=None):
    if previous == desired == reference:
        return reference
    model = model or joint_model(joint=joint)
    flex, spread = joint_angles(desired, reference, model)
    target = (max(model.minimum, min(model.maximum, flex)), max(-model.lateral, min(model.lateral, spread)))
    before = joint_angles(previous, reference, model)
    # Orthogonal flex/spread axes give angular path speed hypot(flex', spread').
    # This bounds the entire interpolation, not just the endpoint quaternion gap.
    distance = math.dist(before, target)
    maximum = (MCP_SPEED if joint == 0 else DISTAL_SPEED)*max(0., dt)
    u = min(1., maximum/max(distance, 1e-15))
    return joint_pose(reference, model, tuple(a+(b-a)*u for a, b in zip(before, target)))


def apply(q, vector):
    return multiply(multiply(q, (0., *vector)), conjugate(q))[1:]


def between(a, b):
    length_a, length_b = max(1e-15, math.hypot(*a)), max(1e-15, math.hypot(*b))
    a = (a[0]/length_a, a[1]/length_a, a[2]/length_a)
    b = (b[0]/length_b, b[1]/length_b, b[2]/length_b)
    dot = a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    cross = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    if dot < -.999999:
        axis = (1., 0., 0.) if abs(a[0]) < .9 else (0., 1., 0.)
        return (0., *normalized((0., a[1]*axis[2]-a[2]*axis[1], a[2]*axis[0]-a[0]*axis[2], a[0]*axis[1]-a[1]*axis[0]))[1:])
    return normalized((1+dot, *cross))


def matrix_quaternion(m):
    """Stable conversion, including 180-degree rotations."""
    candidates = (1+m[0][0]+m[1][1]+m[2][2], 1+m[0][0]-m[1][1]-m[2][2],
                  1-m[0][0]+m[1][1]-m[2][2], 1-m[0][0]-m[1][1]+m[2][2])
    i = max(range(4), key=candidates.__getitem__)
    s = 2 * math.sqrt(max(1e-15, candidates[i]))
    if i == 0:
        return normalized((s/4, (m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s))
    j, k = (i % 3), ((i+1) % 3)
    axis = i-1
    xyz = [0., 0., 0.]
    xyz[axis], xyz[j], xyz[k] = s/4, (m[axis][j]+m[j][axis])/s, (m[axis][k]+m[k][axis])/s
    return normalized(((m[k][j]-m[j][k])/s, *xyz))


def quaternion_matrix(q):
    columns = [apply(q, v) for v in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
    return tuple(tuple(columns[j][i] for j in range(3)) for i in range(3))


def chain_rotations(rest, desired):
    parent, rotations = IDENTITY, []
    for i in range(3):
        segment = tuple(b-a for a, b in zip(rest[i], rest[i+1]))
        target = tuple(b-a for a, b in zip(desired[i], desired[i+1]))
        q = between(segment, apply(conjugate(parent), target))
        rotations.append(q)
        parent = multiply(parent, q)
    return rotations


def forward_chain(rest, rotations):
    result, parent = [rest[0]], IDENTITY
    for i, q in enumerate(rotations):
        parent = multiply(parent, q)
        segment = apply(parent, tuple(b-a for a, b in zip(rest[i], rest[i+1])))
        result.append(tuple(a+b for a, b in zip(result[-1], segment)))
    return result
