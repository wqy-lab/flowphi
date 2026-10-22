"""Bounded, top-down silhouette estimates of upcoming note visibility.

This is a soft camera/readability cost, never a judgement or anatomy constraint.
Visual samples are shared across candidates; no IK is run by the contact scorer.
"""
import bisect
import math
from collections import OrderedDict

from flowphi.charts.model import NoteType
from flowphi.motion import point_at, world_xy

LOOKAHEAD = .65
STEP = .10
MAX_CACHE = 2048


def _remember(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > MAX_CACHE:
        cache.popitem(last=False)
    return value


def _distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    u = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/max(1e-15, dx*dx+dy*dy)))
    return math.hypot(p[0]-a[0]-u*dx, p[1]-a[1]-u*dy)


def _inside(p, polygon):
    inside = False
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        if (a[1] > p[1]) != (b[1] > p[1]):
            if p[0] < a[0]+(p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1]):
                inside = not inside
    return inside


def _hull(points):
    # Handedness reverses the MCP order; do not turn a palm into a bow tie.
    points = sorted(set(tuple(p[:2]) for p in points))
    if len(points) < 3:
        return points
    def half(rows):
        result = []
        for p in rows:
            while len(result) >= 2:
                a, b = result[-2:]
                if (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) > 0:
                    break
                result.pop()
            result.append(p)
        return result
    return half(points)[:-1]+half(reversed(points))[:-1]


class OcclusionScorer:
    def __init__(self, planner):
        self.planner = planner
        self.visuals = {}
        for line in planner.chart.lines:
            for visual in line.visual_notes:
                if not visual.is_fake:
                    self.visuals[id(visual.note)] = visual
        self.tasks = sorted(planner.original_tasks.values(), key=lambda t: t.note.seconds)
        self.beats = [t.note.seconds for t in self.tasks]
        self.holds = [t for t in self.tasks if t.note.type == NoteType.HOLD]
        self.samples, self.contacts = OrderedDict(), OrderedDict()

    def notes(self, t):
        # Only the soft estimate is quantised. Independent contact checks retain
        # their original precision. Same-time geometry is reused by all fingers.
        tick = round(t/.05)
        if tick in self.samples:
            self.samples.move_to_end(tick)
            return self.samples[tick]
        t = tick*.05
        lo, hi = bisect.bisect_right(self.beats, t+.035), bisect.bisect_right(self.beats, t+LOOKAHEAD)
        tasks = self.tasks[lo:hi]+[h for h in self.holds if h.note.seconds <= t+.035 < h.note.seconds+h.note.hold]
        result = []
        p = self.planner
        for task in tasks:
            visual = self.visuals.get(id(task.note))
            if visual is None:
                continue
            state = task.line.note_state(t, visual)
            if state is None:
                continue
            angle = task.line.angle @ t
            along = complex(math.cos(angle), math.sin(angle))
            width = p.screen.width*.1234375*abs(state.width)
            points = [state.head+along*width*u for u in (-.4, 0., .4)]
            if task.note.type == NoteType.HOLD:
                body = [state.head+(state.tail-state.head)*u for u in (.33, .67, 1.)]
                points = points+body if task.note.seconds > t+.035 else body
            visible = [world_xy((v.real/p.screen.width, v.imag/p.screen.height), p.physical)
                       for v in points if 0 <= v.real <= p.screen.width and 0 <= v.imag <= p.screen.height]
            if visible:
                urgency = 1.-.7*max(0., task.note.seconds-t)/LOOKAHEAD
                result.append((visible, urgency*state.alpha))
        return _remember(self.samples, tick, result)

    def coverage(self, t, capsules, palms=()):
        blocked = 0.
        for points, weight in self.notes(t):
            hidden = sum(any(_inside(point, poly) for poly in palms) or
                         any(_distance(point, a, b) <= radius for a, b, radius in capsules)
                         for point in points)
            blocked += weight*hidden/len(points)
        return blocked

    def contact(self, record):
        key = (record['hand'], record['finger'], record['start'], record['end'], id(record['points']))
        if key in self.contacts and self.contacts[key][0] is record['points']:
            self.contacts.move_to_end(key)
            return self.contacts[key][1]
        p = self.planner
        side = -1 if record['hand'] == 'left' else 1
        data = p.profile['fingers'][f"{side}:{record['finger']}"]
        scale = p.profile['scale']/.27
        duration = max(0., record['end']-record['start'])
        count = max(1, math.ceil(duration/STEP))
        total = 0.
        for i in range(count):
            t = record['start']+duration*(i+.5)/count
            tip = world_xy(point_at(record['points'], t), p.physical)
            root = tuple(tip[j]-data['offset'][j] for j in (0, 1))
            base = tuple(root[j]+data['base'][j] for j in (0, 1))
            # A broad wrist-to-knuckle capsule plus the striking finger. Actual
            # ten-finger/palm silhouettes are evaluated during physical rollout.
            palm_end = (root[0], root[1]+.055*scale)
            capsules=[(root,palm_end,.030*scale),(base,tip,.006*scale)]
            # Idle fingers remain visible bodies, even with only index enabled.
            for name in ('thumb','middle','ring','little') if record['finger']=='index' else ('thumb','index','middle','ring','little'):
                if name==record['finger']:
                    continue
                other=p.profile['fingers'][f'{side}:{name}']
                base_other=tuple(root[j]+other['base'][j] for j in (0,1))
                tip_other=tuple(root[j]+other['offset'][j] for j in (0,1))
                capsules.append((base_other,tip_other,.006*scale))
            total += self.coverage(t,capsules)
        result = total*duration/count
        _remember(self.contacts, key, (record['points'], result))
        return result

    def frame(self, t, frame):
        palms, capsules = [], []
        scale = self.planner.profile['scale']/.27
        for hand in frame['hands'].values():
            palms.append(_hull(hand['palm']))
            for finger in hand['fingers'].values():
                joints = finger['joints']
                capsules.extend((a[:2], b[:2], .006*scale) for a, b in zip(joints, joints[1:]))
        return self.coverage(t, capsules, palms)


def scorer(planner):
    if getattr(planner, '_occlusion_scorer', None) is None:
        planner._occlusion_scorer = OcclusionScorer(planner)
    return planner._occlusion_scorer
