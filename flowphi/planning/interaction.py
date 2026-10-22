"""Soft hand interaction costs, independent of wrist order and 3D collisions.

Fast search uses yawed palm/finger proxies, including recent released touches.
Physical rollouts use realised skeletons. Neither path changes hard limits.
"""
import math
from collections import OrderedDict

from flowphi.motion import point_at, world_xy
from flowphi.planning.occlusion import _hull, _inside

PROXY_STEP = .05
RECENT_SECONDS = .30
MAX_CACHE = 1024


def intersects(a, b, c, d):
    """Proper projected crossings, including fingers separated in height."""
    u = b[0]-a[0], b[1]-a[1]
    v = d[0]-c[0], d[1]-c[1]
    det = u[0]*v[1]-u[1]*v[0]
    if abs(det) < 1e-12:
        return False
    w = c[0]-a[0], c[1]-a[1]
    s = (w[0]*v[1]-w[1]*v[0])/det
    t = (w[0]*u[1]-w[1]*u[0])/det
    return .02 < s < .98 and .02 < t < .98


def relation(left, right, scale=1.):
    """Return independent crossing, palm intrusion, tip inversion and twist."""
    crossing, intrusion, inversion, twist = 0., 0., 0., 0.
    if not left['fingers'] or not right['fingers']:
        return crossing, intrusion, inversion, twist
    for a in left['fingers'].values():
        for b in right['fingers'].values():
            weight = min(a['engagement'], b['engagement'])
            if weight <= .01:
                continue
            first, second = a['joints'], b['joints']
            if any(intersects(p, q, r, s) for p, q in zip(first, first[1:])
                   for r, s in zip(second, second[1:])):
                crossing = max(crossing, weight)
            depth = first[-1][0]-second[-1][0]-.008*scale
            inversion = max(inversion, weight*min(4., max(0., depth)/(.025*scale)))
    for hand, other in ((left, right), (right, left)):
        opposite_engagement = max(v['engagement'] for v in other['fingers'].values())
        for value in hand['fingers'].values():
            joints = value['joints']
            weight = min(value['engagement'], opposite_engagement)
            # Include shafts rather than just the final pad, while excluding
            # the MCP endpoint shared with a palm's boundary.
            points = [tuple(a[k]+(b[k]-a[k])*u for k in (0, 1))
                      for a, b in zip(joints, joints[1:]) for u in (.35, .7, 1.)]
            inside = sum(_inside(p, other['palm']) for p in points)
            intrusion = max(intrusion, weight*inside/max(1, len(points)))
            rotation = hand.get('rotation')
            if rotation is not None:
                direction = tuple(joints[-1][k]-joints[0][k] for k in range(3))
                lateral = sum(direction[k]*rotation[k][0] for k in range(3))
                forward = sum(direction[k]*rotation[k][1] for k in range(3))
                angle = max(0., math.atan2(abs(lateral), max(.005, forward))-.45)
                twist = max(twist, value['engagement']*angle)
    return crossing, intrusion, inversion, twist


def density(values):
    crossing, intrusion, inversion, twist = values
    return 8.*crossing+12.*intrusion+6.*inversion+4.*twist


def frame_relation(frame, scale=1.):
    hands = {}
    for name, hand in frame['hands'].items():
        hands[name] = dict(palm=_hull(hand['palm']), rotation=hand['rotation'], fingers={
            finger: dict(joints=value['joints'], engagement=min(1., value.get('influence', float(value['down']))))
            for finger, value in hand['fingers'].items()})
    return relation(hands['left'], hands['right'], scale)


def _proxy(planner, records, t, previous=None):
    fingers, palms = {}, {}
    for record in records:
        if record['start'] > t:
            gap = record['start']-t
            if gap > .25:
                continue
            point = record['points'][0][1:]
            engagement = 1.-gap/.25
            if previous is not None and record['start'] > previous['end'] and t >= previous['end']:
                u = (t-previous['end'])/(record['start']-previous['end'])
                point = tuple(a+(b-a)*u for a, b in zip(previous['points'][-1][1:], point))
                engagement = 1.
        elif t >= record['end']:
            age = t-record['end']
            if age >= RECENT_SECONDS:
                continue
            point, engagement = record['points'][-1][1:], 1.-age/RECENT_SECONDS
        else:
            point, engagement = point_at(record['points'], t), 1.
        # Prefer the most engaged event of a finger, never duplicate its body.
        finger = record['finger']
        if finger in fingers and fingers[finger]['engagement'] >= engagement:
            continue
        tip = world_xy(point, planner.physical)
        side = -1 if record['hand'] == 'left' else 1
        yaw = max(-.45, min(.45, -math.atan2(tip[0]-side*.12, max(.08, tip[1]+.24))))
        c, s = math.cos(yaw), math.sin(yaw)
        def rotate(v):
            return c*v[0]-s*v[1], s*v[0]+c*v[1]
        data = planner.profile['fingers'][f'{side}:{finger}']
        offset, base = rotate(data['offset']), rotate(data['base'])
        root = tip[0]-offset[0], tip[1]-offset[1]
        mcp = root[0]+base[0], root[1]+base[1]
        factor = planner.profile['scale']/.27
        palm = [(root[0]+x*c-y*s, root[1]+x*s+y*c)
                for x, y in ((-.035*factor,.012*factor), (.035*factor,.012*factor),
                             (-.035*factor,.065*factor), (.035*factor,.065*factor))]
        fingers[finger] = dict(joints=[(*mcp, 0.), (*tip, 0.)], engagement=engagement)
        palms[finger] = palm
    return dict(fingers=fingers, palm=_hull([p for polygon in palms.values() for p in polygon]))


def contact_relation(planner, record, history):
    """Incremental search estimate with finite sampling and identity-safe LRU."""
    # The latest per-key predecessors suffice outside the current interval;
    # this includes old live Holds, but not hundreds of finished old touches.
    low, high = record['start']-.25, record['end']
    opposite, seen, previous = [], set(), None
    needed = set(planner.settings.fingers)
    for other in reversed(history):
        if other['hand'] == record['hand']:
            if previous is None and other['finger'] == record['finger'] and other['end'] <= record['start']:
                previous = other
        elif other['start'] <= high+.25:
            if other['end'] > low-RECENT_SECONDS or other['finger'] not in seen:
                opposite.append(other)
            if other['start'] <= low:
                seen.add(other['finger'])
        if previous is not None and seen >= needed:
            break
    if not opposite:
        return dict(cost=0., crossing_seconds=0., palm_seconds=0., inversion_seconds=0., peak_inversion=0.)
    if previous is not None:
        low = max(low, previous['end'])
    cache = getattr(planner, '_interaction_cache', None)
    if cache is None:
        cache = planner._interaction_cache = OrderedDict()
    records = [record, *opposite, *([previous] if previous else [])]
    key = tuple((v['hand'], v['finger'], v['start'], v['end'], id(v['points'])) for v in records)
    if key in cache:
        cache.move_to_end(key)
        return cache[key][1]
    count = max(1, min(128, math.ceil((high-low)/PROXY_STEP)))
    dt = max(0., high-low)/count
    integrals, peak = [0., 0., 0., 0.], 0.
    for i in range(count):
        t = low+(i+.5)*dt
        own = _proxy(planner, (record,), t, previous)
        other = _proxy(planner, opposite, t)
        a, b = (own, other) if record['hand'] == 'left' else (other, own)
        values = relation(a, b, planner.profile['scale']/.27)
        integrals = [total+value*dt for total, value in zip(integrals, values)]
        peak = max(peak, values[2])
    # A quick wrong-side tap must not be almost free just because its DOWN is
    # short. Sustained pricing alone lets entire alternating streams swap hands.
    peak_cost=8.*peak*peak
    report = dict(cost=density(integrals)+peak_cost, peak_order_cost=peak_cost,
                  crossing_seconds=integrals[0], palm_seconds=integrals[1],
                  inversion_seconds=integrals[2], peak_inversion=peak,
                  model='yawed_palm_finger_with_transfers')
    cache[key] = (tuple(v['points'] for v in records), report)
    if len(cache) > MAX_CACHE:
        cache.popitem(last=False)
    return report
