"""Evaluate isolated fingering candidates and retain only revision-bound results.

All validation covers the entire chart, including unmodified boundary contacts.
An internal pass never claims native-game AP or Blender mesh validation.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import copy
import hashlib
import json
import math
import time
import uuid

from flowphi.services.operations import apply_operations
from flowphi.ui.model import PlanDocument


def fingerprint(document):
    raw = json.dumps(document.snapshot(), ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def movement_metrics(plan):
    """Planar finger travel proxies, not anatomical reach or naturalness scores."""
    width, height = plan['physical_screen']
    def distance(a, b):
        return math.hypot((a[0] - b[0]) * width, (a[1] - b[1]) * height)
    path, transfers, peak_speed = 0., 0., 0.
    previous, edges = {}, []
    for contact in sorted(plan['contacts'], key=lambda c: (c['start'], c['end'])):
        points = contact['points']
        path += sum(distance(a[1:], b[1:]) for a, b in zip(points, points[1:]))
        key = contact['hand'], contact['finger']
        if key in previous:
            old = previous[key]
            travel = distance(old['points'][-1][1:], points[0][1:])
            transfers += travel
            gap = contact['start'] - old['end']
            if gap > 0:
                peak_speed = max(peak_speed, travel / gap)
        previous[key] = contact
        edges.extend(((contact['start'], 1), (contact['end'], -1)))
    active, peak = 0, 0
    for _, change in sorted(edges):
        active += change
        peak = max(peak, active)
    return dict(contact_path_m=round(path, 6), transfer_distance_m=round(transfers, 6),
                total_finger_travel_m=round(path + transfers, 6),
                peak_transfer_speed_lower_bound_m_s=round(peak_speed, 6),
                peak_simultaneous_contacts=peak,
                interpretation='编译触控的平面行程与接续速度下界；不含抬指、手腕或皮肤运动')


def validate_compiled(document, chart, *, time_range=None, progress=None):
    """Validate exactly the touch data produced by the existing export compiler."""
    from flowphi.ui.compiler import compile_plan
    from flowphi.ui.validation import quick_check, validate

    started = time.monotonic()
    problems = quick_check(document, chart=chart, time_range=time_range)
    if problems:
        return dict(valid=False, issues=problems, geometry_checked=False,
                    native_ap_validated=False, time_range=time_range, validation_seconds=time.monotonic()-started)
    raw, plan = compile_plan(document, chart, time_range=time_range, check=True)
    canonical = PlanDocument(plan, chart=document.chart, copy_baseline=False, allow_empty=True)
    report = validate(canonical, chart, time_range=time_range, resolved=True, progress=progress)
    for problem in report['issues']:
        problem['contacts'] = sorted({j for i in problem['contacts']
                                     for j in plan['contacts'][i]['editor_contacts']})
    report['validated_psap_sha256'] = hashlib.sha256(raw).hexdigest()
    report['movement'] = movement_metrics(plan)
    report['validation_seconds'] = time.monotonic()-started
    return report


def checked_report(document, chart, *, progress=None):
    try:
        report = validate_compiled(document, chart, progress=progress)
    except (ValueError, OverflowError) as exc:
        report = dict(valid=False, geometry_checked=False, issues=[dict(
            code='compile_error', message=str(exc), contacts=[], severity='error', time=0.)])
    report['native_ap_validated'] = False
    report['mesh_checked'] = False
    report['scope'] = 'full_chart'
    return report


def summary(report, offset=0.):
    issues = report.get('issues', [])
    rows = []
    for problem in issues[:50]:
        row = copy.deepcopy(problem)
        row['contacts'] = [i + 1 for i in row.get('contacts', [])]
        for key in ('time', 'end_time'):
            if key in row:
                row[key] += offset
        rows.append(row)
    return dict(valid=bool(report['valid']), geometry_checked=bool(report.get('geometry_checked')),
                errors=sum(p.get('severity', 'error') == 'error' for p in issues),
                warnings=sum(p.get('severity') == 'warning' for p in issues),
                issue_counts=dict(Counter(p['code'] for p in issues)), issues=rows,
                omitted_issues=max(0, len(issues) - len(rows)),
                timebase='music_seconds', contact_ids='one_based', scope='full_chart',
                native_ap_validated=False, mesh_checked=False,
                movement=report.get('movement'),
                validated_psap_sha256=report.get('validated_psap_sha256'))


def evaluate_candidate(document, chart, operations, *, offset=0., selected_ids=(),
                       fallback_range=None, progress=None):
    """No writes, preferences, live mutations, network or rendering side effects."""
    if not isinstance(offset, (int, float)) or isinstance(offset, bool):
        raise ValueError('音乐偏移无效')
    if not math.isfinite(offset):
        raise ValueError('音乐偏移无效')
    base = fingerprint(document)
    candidate = apply_operations(document, operations, chart=chart, offset=offset,
                                 selected_ids=selected_ids, fallback_range=fallback_range)
    baseline = checked_report(document, chart, progress=progress)
    report = checked_report(candidate, chart, progress=progress)
    before, after = summary(baseline, offset), summary(report, offset)
    travel_delta = (round(after['movement']['total_finger_travel_m'] - before['movement']['total_finger_travel_m'], 6)
                    if after['movement'] is not None and before['movement'] is not None else None)
    changed = [i + 1 for i in range(len(candidate)) if candidate.contact(i) != document.contact(i)]
    return dict(base_fingerprint=base, candidate_fingerprint=fingerprint(candidate),
                snapshot=copy.deepcopy(candidate.snapshot()), report=report,
                result=dict(eligible=bool(report['valid'] and report.get('geometry_checked')),
                    changed_contacts=changed, pose_changed=candidate.pose != document.pose,
                    baseline=before, candidate=after,
                    comparison=dict(error_delta=after['errors'] - before['errors'],
                                    warning_delta=after['warnings'] - before['warnings'],
                                    total_finger_travel_delta_m=travel_delta),
                    limitations=['内部触控检查，不代表官方游戏 AP', '未进行 Blender 皮肤接触、碰撞或自然度评估']))


class CandidateStore:
    """Bounded session-local candidates; no candidate supplied by a model is trusted."""
    def __init__(self, capacity=8):
        if type(capacity) is not int or not 1 <= capacity <= 32:
            raise ValueError('候选容量应为 1–32')
        self.capacity = capacity
        self._entries = OrderedDict()

    def remember(self, evaluation, document, *, revision, offset=0.):
        if revision != document.revision or evaluation['base_fingerprint'] != fingerprint(document):
            raise ValueError('stale：评估期间工程已改变，请重新评估')
        # Snapshot integrity and successful checks remain private to this store.
        if fingerprint(PlanDocument.from_dict(copy.deepcopy(evaluation['snapshot']))) != evaluation['candidate_fingerprint']:
            raise ValueError('候选内容与评估指纹不一致')
        identity = uuid.uuid4().hex
        self._entries[identity] = dict(evaluation=copy.deepcopy(evaluation), document=document,
                                      revision=revision, offset=offset)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return dict(candidate_id=identity, **copy.deepcopy(evaluation['result']))

    def prepare_commit(self, identity, document, *, offset=0.):
        if not isinstance(identity, str) or identity not in self._entries:
            raise ValueError('候选不存在或已过期，请重新评估')
        entry = self._entries[identity]
        evaluation = entry['evaluation']
        if (entry['document'] is not document or entry['revision'] != document.revision
                or entry['offset'] != offset or evaluation['base_fingerprint'] != fingerprint(document)):
            raise ValueError('stale：当前工程或音乐偏移已改变，请重新评估')
        if not evaluation['result']['eligible']:
            raise ValueError('候选未通过完整内部检查，不能采用；请修正后重新评估')
        candidate = PlanDocument.from_dict(copy.deepcopy(evaluation['snapshot']))
        if fingerprint(candidate) != evaluation['candidate_fingerprint']:
            raise ValueError('候选内容与评估指纹不一致')
        return candidate, dict(candidate_id=identity, **copy.deepcopy(evaluation['result']))

    def discard(self, identity):
        self._entries.pop(identity, None)
