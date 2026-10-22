"""Compare two-index chords atomically before committing either hand.

Hand order uses actual contact paths, not line IDs, note order or wrist roots.
Coarse transfer estimates may be replaced only by a successful hard rollout.
"""
import copy
from collections import Counter

from flowphi.charts.model import NoteType
from flowphi.motion import point_at

MAX_ROLLOUTS = 4


def ordered(contacts, ids, physical):
    pair = [c for c in contacts if ids.intersection(c['note_ids'])]
    if len(pair)==1 and ids <= set(pair[0]['note_ids']):
        return True
    left = [c for c in pair if c['hand'] == 'left']
    right = [c for c in pair if c['hand'] == 'right']
    if len(left) != 1 or len(right) != 1:
        return False
    a, b = left[0], right[0]
    low, high = max(a['start'], b['start']), min(a['end'], b['end'])
    if high <= low:
        return False
    times = {low, (low+high)/2, high-1e-7}
    times.update(t for c in pair for t, _, _ in c['points'] if low <= t < high)
    return all((point_at(a['points'], t)[0]-point_at(b['points'], t)[0])*physical[0] <= .008
               for t in times)


def chord_group(tasks, index, enabled_fingers):
    if tuple(enabled_fingers) != ('index',):
        return None
    task = tasks[index]
    beat=getattr(task,'beat',task.start)
    if index and getattr(tasks[index-1],'beat',tasks[index-1].start) == beat:
        return None
    group = []
    for other in tasks[index:index+4]:
        if other.start-task.start>35 or getattr(other,'beat',other.start)!=beat:
            break
        group.append(other)
    if len(group) not in (2,3) or not any(t.note.type in (NoteType.TAP,NoteType.HOLD) for t in group):
        return None
    if len(group)==3 and (sum(t.note.type in (NoteType.FLICK,NoteType.DRAG) for t in group)!=1 or
                          sum(t.note.type in (NoteType.TAP,NoteType.HOLD) for t in group)!=2):
        return None
    return group


def chord_states(planner, group, incoming, progress=None, *, width=None):
    from flowphi.planning.planner import State
    ids = {t.id for t in group}
    # A chord does not enlarge the outer budget; also respect smaller callers.
    width = planner.settings.beam_width if width is None else min(width, planner.settings.beam_width)
    diagnostics = getattr(planner, 'chord_diagnostics', None)
    if diagnostics is None:
        diagnostics = planner.chord_diagnostics = Counter()
    diagnostics['groups'] += 1

    def expand(degraded):
        values = []
        # Both visitation orders prevent the first note's stationary hand from
        # taking all beam slots before the other note has been considered.
        if len(group)==3:
            reusable=next(t for t in group if t.note.type in (NoteType.FLICK,NoteType.DRAG))
            heads=[t for t in group if t is not reusable]
            orders=(((reusable,*heads),(reusable,*heads[::-1])) if reusable.note.type==NoteType.FLICK else
                    ((*heads,reusable),(*heads[::-1],reusable)))
        else:
            orders=(group,group[::-1])
        physical=getattr(planner,'_physical_plain_search',False)
        for state in (incoming[:4] if physical else incoming):
            for order in orders:
                frontier=[state]
                for position,task in enumerate(order):
                    expanded=[]
                    for previous in frontier:
                        if task.note.type==NoteType.DRAG:
                            expanded.extend(planner.share_drag(task,previous))
                        for cost,record in planner.choices(task,previous,degraded=degraded):
                            expanded.append(State((*previous.contacts,record),previous.score+cost))
                        if any(c['kind']=='flick' and ids.intersection(c['note_ids']) for c in previous.contacts):
                            shared=planner.share_flick_hold(task,previous)
                            if shared:
                                from flowphi.planning.scoring import score_history
                                for value in shared:
                                    scored=score_history(planner,value.contacts)
                                    errors={v for c in scored.contacts if ids.intersection(c['note_ids']) for v in c['decision']['degraded']}
                                    if degraded or not errors:
                                        expanded.append(scored)
                    frontier=(sorted(expanded,key=lambda s:s.score)[:4] if physical else
                              expanded if position==len(order)-1 else
                              sorted(expanded,key=lambda s:s.score)[:max(8,width)])
                for value in frontier:
                    current=[c for c in value.contacts if ids.intersection(c['note_ids'])]
                    if ids <= {n for c in current for n in c['note_ids']}:
                        values.append(State(tuple(sorted(value.contacts,key=lambda c:(c['start'],c['pointer']))),value.score))
        unique = {}
        for state in values:
            key = tuple((c['hand'], tuple(c['note_ids']), tuple(map(tuple, c['points'])))
                        for c in state.contacts if ids.intersection(c['note_ids']))
            # Prefix identity matters: the same chord may have different
            # preparations or an ongoing Hold in another incoming state.
            prefix = tuple(id(c) for c in state.contacts if not ids.intersection(c['note_ids']))
            token = prefix, key
            if token not in unique or state.score < unique[token].score:
                unique[token] = state
        return sorted(unique.values(), key=lambda state: state.score)

    normal = expand(False)
    straight = [s for s in normal if ordered(s.contacts, ids, planner.physical)]
    if getattr(planner,'_physical_plain_search',False):
        # The enclosing ordinary window will compare complete physical
        # rollouts, including the lead-in and following independent DOWNs.
        return (straight+[s for s in normal if not ordered(s.contacts,ids,planner.physical)])[:width]
    if straight:
        diagnostics['ordered'] += 1
        return straight[:width]

    # Do not accept other hard violations just to uncross. A larger transfer
    # gets its own exact, bounded check rather than a higher speed allowance.
    relaxed, crossed = [], []
    for state in expand(True):
        current = [c for c in state.contacts if ids.intersection(c['note_ids'])]
        errors = {v for c in current for v in c['decision']['degraded']}
        if errors and errors <= {'finger_transfer', 'wrist_transfer'}:
            (relaxed if ordered(state.contacts, ids, planner.physical) else crossed).append(state)
    # If neither assignment passes coarse checks, compare both complete drafts
    # under the same hard solver instead of falling back to greedy single notes.
    trials = relaxed[:MAX_ROLLOUTS] if normal else (
        relaxed[:MAX_ROLLOUTS//2]+crossed[:MAX_ROLLOUTS//2])
    drafts = []
    if trials:
        from flowphi.planning.hand_evaluation import HandEvaluator
        evaluator = getattr(planner, '_hand_evaluator', None)
        if evaluator is None:
            evaluator = planner._hand_evaluator = HandEvaluator(planner)
        low = min(t.start for t in group)/1000-.25
        high = max(c['end'] for c in trials[0].contacts if ids.intersection(c['note_ids']))
        context = [t for t in planner.tasks if t.start/1000 <= high and t.end/1000 >= low
                   and (t.start <= group[0].start or t.id in ids)]
        for state in trials:
            diagnostics['rollouts'] += 1
            motion = evaluator.evaluate(state.contacts, context,
                (lambda done, total: progress(0, 1, group[0])) if progress else None,
                time_range=(low, high))
            if not motion['valid']:
                drafts.append((state,motion))
                continue
            contacts, refund = [], 0.
            for c in state.contacts:
                if ids.intersection(c['note_ids']):
                    c = copy.deepcopy(c)
                    decision = c['decision']
                    refund += decision['costs']['degraded']
                    decision['costs']['degraded'] = 0.
                    decision['degraded'] = []
                    decision['verified_transfer'] = motion
                contacts.append(c)
            diagnostics['verified_ordered' if ordered(state.contacts,ids,planner.physical) else 'verified_crossed'] += 1
            return [State(tuple(contacts), state.score-refund)]
    if not normal and planner.settings.allow_degraded and drafts:
        def severity(item):
            state,motion=item
            return ((motion.get('max_error_mm',1e3)/4.)**2+
                    (motion.get('max_height_error_mm',1e3)/2.)**2+
                    (motion.get('skeleton_penetration_mm',1e3)/3.)**2,
                    not ordered(state.contacts,ids,planner.physical),state.score)
        state,motion=min(drafts,key=severity)
        contacts=[]
        for c in state.contacts:
            if ids.intersection(c['note_ids']):
                c=copy.deepcopy(c)
                c['decision']['spatial_chord_draft']=motion
            contacts.append(c)
        diagnostics['failed_complete_drafts']+=1
        return [State(tuple(contacts),state.score)]
    diagnostics['crossed_or_unavailable'] += 1
    return normal[:width]
