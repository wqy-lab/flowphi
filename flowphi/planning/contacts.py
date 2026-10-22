from functools import lru_cache
from collections import OrderedDict
import cmath
import math

import numpy as np
import shapely
from shapely.affinity import scale, translate
from shapely.geometry import Point, box
from shapely.ops import nearest_points, unary_union

from flowphi.planning.geometry import JudgeArea, JUDGE_HALF_DRAG, JUDGE_HALF_TAP
from flowphi.charts.model import NoteType
from flowphi.charts.blocks import compose
from flowphi.motion import point_at

SCREEN_MARGIN = .012


class ContactGeometry:
    def __init__(self, chart, mask_size=(196, 110), pad_clearance=.006, boundary_mode='aligned',
                 physical_screen=(.28, .1575), extra_clearance_m=0., cache_judgement=False):
        self.chart = chart
        self.notes = dict(enumerate((line, n) for line in chart.lines for n in line.notes))
        self.screen = box(SCREEN_MARGIN, SCREEN_MARGIN, 1-SCREEN_MARGIN, 1-SCREEN_MARGIN).difference(
            box(0, 0, .1, .1).union(box(.9, 0, 1, .1)))
        # Normalized shader noise is in [-.5,.5], along two orthogonal axes.
        # Include a mask texel and an effect texel for quantization/edge dilation.
        # Soft decorative glow is not treated as the filled blocking region.
        if boundary_mode not in ('aligned', 'native'):
            raise ValueError(f'Unknown block boundary mode: {boundary_mode}')
        self.display_margin = (math.sqrt(.5) * .1 if boundary_mode == 'native' else 0.) + 1.5 / min(mask_size) + pad_clearance
        self.physical_screen = tuple(physical_screen)
        self.extra_clearance_m = float(extra_clearance_m)
        if not math.isfinite(self.extra_clearance_m) or self.extra_clearance_m < 0:
            raise ValueError('Block clearance must be finite and nonnegative')
        if len(self.physical_screen) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.physical_screen):
            raise ValueError('Physical screen dimensions must be finite and positive')
        # A planner owns an immutable chart. Bound caches to this geometry's
        # lifetime; final export validation uses the independent uncached path.
        self.cache_judgement = cache_judgement
        self._sample_cache = OrderedDict()
        if cache_judgement:
            self._strip_at = lru_cache(maxsize=8192)(self._strip_at)
            self._required_zone = lru_cache(maxsize=4096)(self._required_zone)
            self._guarded_zone = lru_cache(maxsize=4096)(self._guarded_zone)

    def normalized(self, geometry):
        return scale(geometry, xfact=1 / self.chart.width, yfact=1 / self.chart.height, origin=(0, 0))

    @lru_cache(maxsize=32768)
    def blocked(self, t, extra_clearance_m=None):
        return self.blocked_rectangles(self.chart.block_areas.active(t), extra_clearance_m)

    @lru_cache(maxsize=8192)
    def blocked_rectangles(self, rectangles, extra_clearance_m=None):
        raw = self.normalized(compose(rectangles))
        # Shapely approximates circular buffers with an inscribed polygon. Inflate
        # by the secant of half an arc step to keep the bound conservative.
        radius = self.display_margin / math.cos(math.pi / 64)
        if raw.is_empty:
            return raw
        guarded = raw.buffer(radius, quad_segs=16)
        extra_clearance_m = self.extra_clearance_m if extra_clearance_m is None else float(extra_clearance_m)
        if not math.isfinite(extra_clearance_m) or extra_clearance_m < 0:
            raise ValueError('Block clearance must be finite and nonnegative')
        if extra_clearance_m:
            w, h = self.physical_screen
            physical = scale(guarded, xfact=w, yfact=h, origin=(0, 0))
            guarded = scale(physical.buffer(extra_clearance_m / math.cos(math.pi / 64), quad_segs=16),
                            xfact=1 / w, yfact=1 / h, origin=(0, 0))
        return guarded

    def judgement_time(self, contact, nid):
        return contact.get('judgement_times', {}).get(str(nid), self.notes[nid][1].seconds)

    def strip(self, line, note, t, ratio, judgement_time=None):
        return self._strip_at(line, note, self._strip_time(note, t, judgement_time), ratio)

    @staticmethod
    def _strip_time(note, t, judgement_time):
        when = max(note.seconds, min(t, note.seconds + note.hold - .001)) if note.type == NoteType.HOLD else (
            note.seconds if judgement_time is None else judgement_time)
        return when

    def _strip_at(self, line, note, when, ratio):
        half = JUDGE_HALF_DRAG if note.type in (NoteType.DRAG, NoteType.FLICK) else JUDGE_HALF_TAP
        return self.normalized(JudgeArea(line.pos(when, note.offset),
            cmath.exp(1j * (line.angle @ when)), self.chart.width, self.chart.height, half, ratio).poly)

    def _required_zone(self, specs, joint):
        strips = [self._strip_at(*spec) for spec in specs]
        zone = self.screen
        if joint:
            for strip in strips:
                zone = zone.intersection(strip)
        else:
            zone = zone.intersection(unary_union(strips))
        return zone

    @staticmethod
    def _guarded_zone(zone):
        return zone.buffer(1e-9)

    def _required(self, contact, t):
        joint = contact.get('joint_note_coverage') or any(
            w['start'] <= t < w['end'] for w in contact.get('joint_note_windows', ()))
        owner = [(*self.notes[nid], self.judgement_time(contact, nid)) for nid in contact['note_ids']]
        if contact.get('collective_hold'):
            if not owner or any(n.type != NoteType.HOLD for _, n, _ in owner):
                raise ValueError('Collective coverage is only valid for Hold notes')
            owner = []
        holds = [(line, n, beat) for line, n, beat in owner if n.type == NoteType.HOLD and
                 n.seconds - .0011 <= t <= n.seconds + n.hold + .0011]
        required = holds or [(line, n, beat) for line, n, beat in owner if abs(t - beat) <= .0011 or
                            beat <= t <= beat + .010]
        if joint:
            required = holds + [(line, n, beat) for line, n, beat in owner if n.type != NoteType.HOLD and
                (abs(t - beat) <= .0011 or beat <= t <= beat + .010)]
        return required, joint

    def _interior_point(self, contact, t, xy):
        """Sufficient interior test for rectangles, with a numerical guard.

        Boundary points and authored region restrictions take the exact polygon
        path. Block clearance is always checked separately by allows_point.
        """
        if contact.get('refinement_region'):
            return False
        x, y = xy
        guard = 1e-7
        if not (.012 + guard < x < .988 - guard and .012 + guard < y < .988 - guard):
            return False
        if y <= .1 + guard and (x <= .1 + guard or x >= .9 - guard):
            return False
        required, joint = self._required(contact, t)
        point = complex(x * self.chart.width, y * self.chart.height)
        margin = guard * max(self.chart.width, self.chart.height)
        def inside(item):
            line, note, beat = item
            when = self._strip_time(note, t, beat)
            center = line.pos(when, note.offset)
            delta = (point - center) * cmath.exp(-1j * (line.angle @ when))
            half = JUDGE_HALF_DRAG if note.type in (NoteType.DRAG, NoteType.FLICK) else JUDGE_HALF_TAP
            ratio = .5 if note.type in (NoteType.HOLD, NoteType.TAP) else .3 if note.type == NoteType.DRAG else .85
            return (abs(delta.real) < self.chart.width * half * ratio - margin and
                    abs(delta.imag) < math.hypot(self.chart.width, self.chart.height) + abs(center) - margin)
        return not required or (all(map(inside, required)) if joint else any(map(inside, required)))

    def zone(self, contact, t, *, central=True, include_blocks=True):
        required, joint = self._required(contact, t)
        # A connected Drag phrase is free to travel BETWEEN judgement windows.
        # Switching the nearest note's strip at its temporal midpoint creates
        # spurious discontinuities, even while the actual note geometry is static.
        # After a Tap's judgement the held finger may travel to a later Drag
        # or Flick. Its old judgement strip is no longer a live constraint.
        zone = self.screen
        if contact.get('refinement_region'):
            zone = zone.intersection(box(*contact['refinement_region']))
        if required:
            specs = tuple((line, n, self._strip_time(n, t, beat),
                (.5 if n.type in (NoteType.HOLD, NoteType.TAP) else
                 .3 if n.type == NoteType.DRAG else .85) if central else .85)
                for line, n, beat in required)
            if self.cache_judgement and not contact.get('refinement_region'):
                zone = self._required_zone(specs, bool(joint))
            else:
                strips = [self._strip_at(*spec) for spec in specs]
                if joint:
                    for strip in strips:
                        zone = zone.intersection(strip)
                else:
                    zone = zone.intersection(unary_union(strips))
        if include_blocks and self.chart.block_areas:
            blocked = self.blocked(t, contact.get('block_clearance_m'))
            if not blocked.is_empty:
                zone = zone.difference(blocked)
        return zone

    def allows_point(self, contact, t, xy):
        interior = self.cache_judgement and self._interior_point(contact, t, xy)
        if not interior and not self._guarded_zone(self.zone(contact, t, include_blocks=False)).covers(Point(*xy)):
            return False
        clearance = contact.get('block_clearance_m', self.extra_clearance_m)
        radius = self.display_margin / math.cos(math.pi / 64)
        margins = [radius + clearance / (s * math.cos(math.pi / 64)) for s in self.physical_screen]
        x, y = xy[0] * self.chart.width, xy[1] * self.chart.height
        nearby = []
        for r in self.chart.block_areas.active(t):
            c, s = abs(math.cos(r.angle)), abs(math.sin(r.angle))
            hx = (r.size[0] * c + r.size[1] * s) / 2 + margins[0] * self.chart.width
            hy = (r.size[0] * s + r.size[1] * c) / 2 + margins[1] * self.chart.height
            if abs(x - r.center[0]) <= hx and abs(y - r.center[1]) <= hy:
                nearby.append(r)
        if not nearby:
            return True
        blocked = self.blocked_rectangles(tuple(nearby), clearance)
        return not blocked.buffer(-1e-9).covers(Point(*xy))

    def times(self, contact, step=.01):
        low, high = round(contact['start'] * 1000), round(contact['end'] * 1000) - 1
        times = {low, high, *range(low, high + 1, max(1, round(step * 1000)))}
        for nid in contact['note_ids']:
            n = self.notes[nid][1]
            beat = round(self.judgement_time(contact, nid) * 1000)
            times.update((beat - 1, beat, beat + 1, beat + 10, beat + 11))
        for t in self.chart.block_areas.key_times:
            ms = t * 1000
            times.update((math.floor(ms) - 1, math.floor(ms), math.ceil(ms), math.ceil(ms) + 1))
        return np.array(sorted(ms / 1000 for ms in times if low <= ms <= high))

    def sample_zones(self, contact, step=.01, extra_times=()):
        if not self.cache_judgement:
            return self._sample_zones(contact, step, extra_times)
        # Sampling depends on geometry and timing, not finger assignment, seed
        # positions or endpoint constraints applied later by refine_contact.
        extra_times = tuple(sorted(extra_times))
        key = (contact['start'], contact['end'], tuple(contact['note_ids']), step, extra_times,
               tuple(sorted(contact.get('judgement_times', {}).items())),
               repr({k: contact.get(k) for k in ('joint_note_coverage', 'joint_note_windows',
                     'collective_hold', 'refinement_region', 'block_clearance_m')}))
        cached = self._sample_cache.get(key)
        if cached is None:
            times, zones = self._sample_zones(contact, step, extra_times)
            cached = (times, tuple(zones))
            self._sample_cache[key] = cached
            if len(self._sample_cache) > 128:
                self._sample_cache.popitem(last=False)
        else:
            self._sample_cache.move_to_end(key)
        # Refinement pins endpoints by replacing list entries. Never let one
        # candidate modify another candidate's cached sample constraints.
        return cached[0].copy(), list(cached[1])

    def _sample_zones(self, contact, step=.01, extra_times=()):
        """Place keys on milliseconds, constraining adjacent keys at exact edges.

        UP happens after the final MOVE's millisecond. Block events can also
        occur between milliseconds, so merely checking the key instants misses
        the last fraction of a dwell or the instant before a block disappears.
        """
        low, high = round(contact['start'] * 1000), round(contact['end'] * 1000) - 1
        checks = [*extra_times, contact['end'] - 1e-7]
        for t in self.chart.block_areas.key_times:
            if contact['start'] <= t < contact['end']:
                checks.extend((t, max(contact['start'], t - 1e-6), min(contact['end'] - 1e-7, t + 1e-6)))
        for nid in contact['note_ids']:
            when = self.judgement_time(contact, nid)
            if contact['start'] - .0011 <= when < contact['end']:
                checks.append(when)
        milliseconds = {round(t * 1000) for t in self.times(contact, step)}
        for t in checks:
            milliseconds.update(max(low, min(high, i)) for i in (math.floor(t * 1000), math.ceil(t * 1000)))
        times = np.array(sorted(ms / 1000 for ms in milliseconds))
        zones = [self.zone(contact, float(t)) for t in times]
        for t in checks:
            legal = self.zone(contact, float(t))
            right = int(np.searchsorted(times, t))
            for i in {max(0, right - 1), min(len(times) - 1, right)}:
                zones[i] = zones[i].intersection(legal)
        return times, [z.buffer(-2e-5) for z in zones]


_PROJECTION_INTERIORS = OrderedDict()


def _projection_data(zone):
    """Certified interior box and exact convex edges; complex shapes fall back."""
    identity = id(zone)
    cached = _PROJECTION_INTERIORS.get(identity)
    if cached is not None and cached[0] is zone:
        _PROJECTION_INTERIORS.move_to_end(identity)
        return cached[1:]
    low_x, low_y, high_x, high_y = zone.bounds
    width, height = high_x-low_x, high_y-low_y
    interiors = []
    singleton=tuple(zone.coords[0][:2]) if isinstance(zone,Point) and not zone.is_empty else None
    if width > 0 and height > 0:
        bounds_list=[(low_x,low_y,high_x,high_y),
            (low_x,low_y+height*.1,high_x,high_y),
            (low_x,low_y,high_x,high_y-height*.1),
            (low_x+width*.1,low_y,high_x-width*.1,high_y)]
        bounds_list.extend((low_x+width*inset,low_y+height*inset,high_x-width*inset,high_y-height*inset)
                           for inset in (.2,.35))
        for bounds in bounds_list:
            if zone.covers(box(*bounds)):
                interiors.append(bounds)
                if bounds==bounds_list[0]:break
    segments, orientation = None, 1.
    if zone.geom_type == 'Polygon' and not zone.interiors and zone.is_valid:
        vertices = list(zone.exterior.coords)
        if 4 <= len(vertices) <= 9:
            edges = [(float(a[0]), float(a[1]), float(b[0]-a[0]), float(b[1]-a[1]))
                     for a, b in zip(vertices, vertices[1:])]
            turns = [a[2]*b[3]-a[3]*b[2] for a, b in zip(edges, [*edges[1:], edges[0]])]
            if (all(t >= 0 for t in turns) or all(t <= 0 for t in turns)) and any(abs(t) > 1e-15 for t in turns):
                orientation = 1. if any(t > 0 for t in turns) else -1.
                if all(dx*dx+dy*dy > 1e-20 for _, _, dx, dy in edges):
                    segments = tuple((*e, e[2]*e[2]+e[3]*e[3], float(b[0]), float(b[1]))
                                     for e, b in zip(edges, vertices[1:]))
    _PROJECTION_INTERIORS[identity] = (zone, tuple(interiors), segments, orientation,singleton)
    if len(_PROJECTION_INTERIORS) > 4096:
        _PROJECTION_INTERIORS.popitem(last=False)
    return tuple(interiors), segments, orientation,singleton


def project(zone, xy):
    interiors, segments, orientation,singleton = _projection_data(zone)
    if singleton is not None:return np.asarray(singleton,dtype=float)
    for interior in interiors:
        if interior[0] <= xy[0] <= interior[2] and interior[1] <= xy[1] <= interior[3]:
            return np.asarray(xy, dtype=float)
    if segments is not None:
        x, y = float(xy[0]), float(xy[1])
        crosses = [orientation*(dx*(y-ay)-dy*(x-ax)) for ax, ay, dx, dy, _, _, _ in segments]
        if all(value > 1e-12 for value in crosses):
            return np.asarray(xy, dtype=float)
        if any(value < -1e-12 for value in crosses):
            best, distance = None, math.inf
            for ax, ay, dx, dy, squared, bx, by in segments:
                u = max(0., min(1., ((x-ax)*dx+(y-ay)*dy)/squared))
                px, py = (ax, ay) if u == 0 else (bx, by) if u == 1 else (ax+u*dx, ay+u*dy)
                candidate = (x-px)**2+(y-py)**2
                if candidate < distance:
                    best, distance = (px, py), candidate
            return np.asarray(best, dtype=float)
    # Same point/region predicate including boundaries, without allocating a
    # Shapely Point for every interior coordinate-descent step.
    if shapely.intersects_xy(zone, xy[0], xy[1]):
        return np.asarray(xy, dtype=float)
    p = Point(*xy)
    q = nearest_points(zone, p)[0]
    return np.array((q.x, q.y))


def refine_contact(geometry, contact, *, iterations=120, extra_times=()):
    times, zones = geometry.sample_zones(contact, extra_times=extra_times)
    empty = [float(t) for t, z in zip(times, zones) if z.is_empty]
    if empty:
        return None, {'reason': 'empty_zone', 'times': empty[:12]}
    for name, index in (('refinement_start', 0), ('refinement_end', len(times) - 1)):
        if contact.get(name) is not None:
            endpoint = Point(*contact[name])
            if not zones[index].buffer(3e-5).covers(endpoint):
                return None, {'reason': 'endpoint_outside_zone', 'time': float(times[index])}
            zones[index] = endpoint
    if contact['kind'] == 'flick':
        allowed = None
        for t, zone in zip(times, zones):
            x, y = point_at(contact['points'], float(t))
            translations = translate(zone, xoff=-x, yoff=-y)
            allowed = translations if allowed is None else allowed.intersection(translations)
            if allowed.is_empty:
                return None, {'reason': 'no_rigid_flick_translation'}
        delta = project(allowed, (0., 0.))
        return [[t, x + delta[0], y + delta[1]] for t, x, y in contact['points']], {
            'stationary': False, 'flick_translation': delta.tolist()}
    shared = zones[0]
    for zone in zones[1:]:
        shared = shared.intersection(zone)
        if shared.is_empty:
            break
    original = np.array([point_at(contact['points'], float(t)) for t in times])
    if len(times) == 1:
        return [[float(times[0]), *project(zones[0], original[0])]], {'stationary': True}
    if not shared.is_empty:
        xy = project(shared, original.mean(axis=0))
        return [[float(times[0]), *xy], [float(times[-1]), *xy]], {'stationary': True}

    # Initialize by minimum motion, then remove the direction bias by sweeping
    # in both directions. Each coordinate update minimizes neighboring travel
    # while remaining inside that instant's legal set.
    xy = original.copy()
    xy[0] = project(zones[0], xy[0])
    for i in range(1, len(xy)):
        xy[i] = project(zones[i], xy[i - 1])
    for sweep in range(iterations):
        before = xy.copy()
        indices = range(len(xy)) if sweep % 2 == 0 else range(len(xy) - 1, -1, -1)
        for i in indices:
            if i == 0:
                target = xy[1]
            elif i == len(xy) - 1:
                target = xy[i - 1]
            else:
                dt0, dt1 = times[i] - times[i - 1], times[i + 1] - times[i]
                target = (xy[i - 1] / dt0 + xy[i + 1] / dt1) / (1 / dt0 + 1 / dt1)
            xy[i] = project(zones[i], target)
        if np.max(np.abs(xy - before)) < 1e-7:
            break
    return [[float(t), *p] for t, p in zip(times, xy)], {'stationary': False, 'sweeps': sweep + 1}


def validate_contact(geometry, contact, points, step=.001, limit=10):
    bad = []
    times = set(geometry.times(contact, step))
    for nid in contact['note_ids']:
        note = geometry.notes[nid][1]
        if note.type == NoteType.HOLD:
            first = max(0, math.ceil((contact['start'] - note.seconds) / step))
            last = min(math.ceil(note.hold / step), math.ceil((contact['end'] - note.seconds) / step))
            times.update(note.seconds + i * step for i in range(first, last))
    for t in geometry.chart.block_areas.key_times:
        if contact['start'] <= t < contact['end']:
            times.update((t, max(contact['start'], t - 1e-6), min(contact['end'] - 1e-7, t + 1e-6)))
    times.add(contact['end'] - 1e-7)
    for t in sorted(times):
        xy = point_at(points, float(t))
        if not geometry.allows_point(contact, float(t), xy):
            bad.append(float(t))
            if len(bad) >= limit:
                break
    return bad


def route_contact(geometry, contact, screen, *, extra_times=()):
    times, zones = geometry.sample_zones(contact, .02, extra_times)
    grid = np.array([(x, y) for x in np.linspace(.025, .975, 15) for y in np.linspace(.025, .975, 23)])
    guide = np.array([point_at(contact['points'], float(t)) for t in np.linspace(times[0], times[-1], 17)])
    candidates, predecessors = [], []
    scores = None
    physical = np.array(screen)
    for index, t in enumerate(times):
        zone = zones[index].buffer(-2e-6)
        if zone.is_empty:
            return None, {'reason': 'empty_route_zone', 'time': float(t)}
        interior = grid[shapely.contains_xy(zone, grid[:, 0], grid[:, 1])]
        additions = [project(zone, p) for p in guide]
        components = [zone] if zone.geom_type == 'Polygon' else list(zone.geoms)
        for component in components:
            if component.geom_type != 'Polygon':
                continue
            centroid = component.representative_point()
            additions.append((centroid.x, centroid.y))
            coords = np.array(component.exterior.coords)
            for p in coords[np.linspace(0, len(coords) - 1, min(24, len(coords)), dtype=int)]:
                additions.append(project(zone, p))
        if candidates:
            # Preserve continuity across small event intervals even when no
            # lattice point occupies the changing sliver of legal space.
            for p in candidates[-1][np.argsort(scores)[:24]]:
                additions.append(project(zone, p))
        points = np.unique(np.round(np.concatenate((interior, np.array(additions))), 9), axis=0)
        endpoint = contact.get('refinement_start' if index == 0 else 'refinement_end' if index == len(times) - 1 else '')
        if endpoint is not None:
            if not zone.buffer(1e-5).covers(Point(*endpoint)):
                return None, {'reason': 'endpoint_outside_zone', 'time': float(t)}
            points = np.array([endpoint], dtype=float)
        if len(points) > 150:
            points = points[np.linspace(0, len(points) - 1, 150, dtype=int)]
        if index == 0:
            scores = .001 * np.sum(((points - guide[0]) * physical)**2, axis=1)
            predecessors.append(None)
        else:
            old = candidates[-1]
            dt = float(t - times[index - 1])
            delta = (old[:, None, :] - points[None, :, :]) * physical
            distance2 = np.sum(delta * delta, axis=2)
            valid = np.ones(distance2.shape, dtype=bool)
            for u in (.25, .5, .75):
                middle = old[:, None, :] * (1 - u) + points[None, :, :] * u
                legal = geometry.zone(contact, float(times[index - 1] + u * dt))
                valid &= shapely.contains_xy(legal.buffer(1e-9), middle[:, :, 0], middle[:, :, 1])
            # Squared velocity cost strongly prefers spreading motion over the
            # available interval. Penalize excessive speed separately.
            speed = np.sqrt(distance2) / dt
            cost = scores[:, None] + distance2 / dt + 100 * np.maximum(0, speed - 1.2)**2 * dt
            cost[~valid] = np.inf
            choice = np.argmin(cost, axis=0)
            scores = cost[choice, np.arange(len(points))]
            if not np.isfinite(scores).any():
                return None, {'reason': 'disconnected_route', 'time': float(t)}
            predecessors.append(choice)
        companion = contact.get('refinement_companion')
        if companion and companion['start'] <= t < companion['end']:
            desired = np.array(point_at(companion['points'], float(t))) + companion['offset']
            scores += 120 * np.sum(((points - desired) * physical)**2, axis=1) * (
                float(times[index] - times[index - 1]) if index else .01)
        candidates.append(points)
    selected = int(np.argmin(scores))
    path = []
    for i in range(len(times) - 1, -1, -1):
        path.append([float(times[i]), *candidates[i][selected]])
        if i:
            selected = int(predecessors[i][selected])
    return list(reversed(path)), {'stationary': False, 'routing': 'whole_contact_graph', 'cost': float(np.min(scores))}
