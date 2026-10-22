"""Bounded alternatives for local Flick/Drag and simultaneous-note handoffs.

Only generated, uncommitted suffixes may change. DOWNs, Hold coverage and
completed Flick strokes in the frozen prefix are never reconstructed.
"""
import copy
import math
from collections import defaultdict
from itertools import islice

from flowphi.charts.model import NoteType
from flowphi.motion import point_at,world_xy,FINGER_ORDER
from flowphi.planning.contacts import project,validate_contact
from flowphi.planning.drag_continuation import _refined_tail

PROPOSAL_LIMIT=12
GEOMETRY_LIMIT=32
ROLLOUT_LIMIT=8
ROUNDS=2


def _key(c):
    return c['hand'],c['finger']


def _guard(planner,c,nid):
    t=planner.original_tasks[nid]
    beat=c.get('judgement_times',{}).get(str(nid),t.beat/1000)
    return (t.note.seconds+t.note.hold if t.note.type==NoteType.HOLD else
            t.end/1000 if t.note.type==NoteType.FLICK else beat+.012)


def _end(planner,c,ids):
    return max((planner.original_tasks[n].end+1)/1000 if planner.original_tasks[n].note.type in
               (NoteType.HOLD,NoteType.FLICK) else
               c.get('judgement_times',{}).get(str(n),planner.original_tasks[n].beat/1000)+
               (.025 if planner.original_tasks[n].note.type==NoteType.TAP else .013) for n in ids)


def _kind(planner,ids):
    kinds={planner.original_tasks[n].note.type for n in ids}
    return next(name for typ,name in ((NoteType.FLICK,'flick'),(NoteType.HOLD,'hold'),
        (NoteType.DRAG,'drag'),(NoteType.TAP,'tap')) if typ in kinds)


def _suffix(planner,c,ids,changed_at,frozen_before,*,seed=None):
    """Re-fit after protected judgements; a head seed is for a new head swap."""
    if seed is None and c['end']<frozen_before-1e-9:return None
    g=planner._ribbon_geometry
    result=copy.deepcopy(c)
    result.update(note_ids=sorted(ids,key=lambda n:(planner.original_tasks[n].beat,n)),
        kind=_kind(planner,ids),end=_end(planner,c,ids),joint_note_coverage=True)
    if result['end']<frozen_before-1e-9:return None
    result['judgement_times']={str(n):v for n,v in c.get('judgement_times',{}).items() if n in ids}
    for n in ids:result['judgement_times'].update(planner.timing_metadata(planner.original_tasks[n]).get('judgement_times',{}))
    if seed is not None:
        if c['start']<frozen_before-1e-9:return None
        first=min(ids,key=lambda n:planner.original_tasks[n].start)
        if planner.original_tasks[first].note.type!=NoteType.TAP:return None
        result['beat']=planner.original_tasks[first].beat/1000
        start=math.ceil((_guard(planner,result,first)-1e-9)*1000)/1000
        anchor=project(g.zone(result,c['start']),seed)
        if anchor is None:return None
        prefix=[[c['start'],*anchor]]
    else:
        protected=[n for n in ids if planner.original_tasks[n].beat/1000<changed_at-.0011]
        if not protected:return None
        start=max(frozen_before,max(_guard(planner,c,n) for n in protected))
        start=math.ceil((start-1e-9)*1000)/1000
        if start>changed_at+.0011:return None
        anchor=point_at(c['points'],start)
        prefix=[list(p) for p in c['points'] if p[0]<start]
    if result['end']<=start:
        result['points']=prefix or [[c['start'],*anchor]]
    elif not any(planner.original_tasks[n].beat/1000>=start-.0011 for n in ids):
        result['points']=[*prefix,[start,*anchor]]
    else:
        tail=dict(result,start=start,refinement_start=list(anchor),
            points=[[start,*anchor],[result['end']-.001,*anchor]])
        points=_refined_tail(planner,g,tail)
        if points is None:return None
        result['points']=[*prefix,*points]
    if result['points'][-1][0]>=result['end']:return None
    if validate_contact(g,result,result['points'],limit=1):return None
    if any(math.dist(world_xy(a[1:],planner.physical),world_xy(b[1:],planner.physical))>
           planner.settings.finger_speed*(b[0]-a[0])+1e-9 for a,b in zip(result['points'],result['points'][1:])):return None
    result.pop('terminal_hold_release_ms',None);result.pop('terminal_hold_release_reason',None)
    return result


def _flick_routes(planner,contacts,task,frozen_before):
    """Try a complete directional stroke followed by the actual yellow target."""
    from flowphi.planning.planner import State
    from flowphi.planning.drag_continuation import drag_continuations
    owners={n:i for i,c in enumerate(contacts) for n in c['note_ids']}
    source_index=owners.get(task.id)
    if source_index is None:return
    source=contacts[source_index]
    flicks=[]
    for i,c in enumerate(contacts):
        if c['start']<frozen_before-1e-9 or c['start']>=task.beat/1000:continue
        ids=[n for n in c['note_ids'] if planner.original_tasks[n].note.type==NoteType.FLICK]
        if len(ids)!=1 or any(planner.original_tasks[n].note.type in (NoteType.TAP,NoteType.HOLD) for n in c['note_ids']):continue
        f=planner.original_tasks[ids[0]]
        if abs(c['start']-f.start/1000)>.0011:continue
        if abs(c.get('judgement_times',{}).get(str(f.id),f.beat/1000)-f.beat/1000)>.0011:continue
        if not .001<task.beat/1000-f.beat/1000<=.18:continue
        flicks.append((i,c,f))
    for index,c,flick in flicks[-2:]:
        keep=[n for n in source['note_ids'] if n!=task.id]
        replacements={}
        if index!=source_index:
            if not keep:
                if any(planner.original_tasks[n].note.type!=NoteType.DRAG for n in source['note_ids']):continue
                replacements[source_index]=None
            else:
                short=_suffix(planner,source,keep,task.beat/1000,frozen_before)
                if short is None:continue
                replacements[source_index]=short
        tail_ids=[n for n in c['note_ids'] if n!=flick.id]
        if any(planner.original_tasks[n].note.type!=NoteType.DRAG for n in tail_ids):continue
        ids=set(tail_ids)|{task.id}
        seed=point_at(c['points'],flick.beat/1000)
        values=[]
        for direction in (1,-1,2,-2):
            path=planner.path(flick,(seed[0]*planner.screen.width,seed[1]*planner.screen.height),flick.end,direction)
            if path is None:continue
            stroke=dict(c,note_ids=[flick.id],kind='flick',end=(flick.end+1)/1000,
                        points=path,joint_note_coverage=True)
            base=[replacements.get(i,v) if i in replacements else v for i,v in enumerate(contacts) if i!=index]
            base=[v for v in base if v is not None]
            state=State(tuple(sorted([*base,stroke],key=lambda v:(v['start'],v['pointer']))))
            valid=True
            for nid in sorted(ids,key=lambda n:planner.original_tasks[n].beat):
                choices=drag_continuations(planner,planner.original_tasks[nid],state)
                choices=[s for s in choices if any(flick.id in v['note_ids'] and nid in v['note_ids'] for v in s.contacts)]
                if not choices:valid=False;break
                state=min(choices,key=lambda s:s.score)
            if not valid:continue
            record=next(v for v in state.contacts if flick.id in v['note_ids'])
            distance=sum(math.dist(world_xy(a[1:],planner.physical),world_xy(b[1:],planner.physical))
                         for a,b in zip(record['points'],record['points'][1:]))
            values.append((distance,list(state.contacts)))
        for _,value in sorted(values,key=lambda v:v[0])[:2]:
            yield 'flick_drag_handoff',value


def _simultaneous(planner,contacts,tasks,frozen_before):
    """Swap note ownership independently of a merged contact's earlier DOWN."""
    owners={n:i for i,c in enumerate(contacts) for n in c['note_ids']}
    beats=defaultdict(list)
    for t in tasks:
        if t.note.type in (NoteType.TAP,NoteType.DRAG) and t.id in owners:beats[t.beat].append(t)
    for beat,group in beats.items():
        if len(group)!=2 or group[0].note.type!=group[1].note.type:continue
        a,b=group;i,j=owners[a.id],owners[b.id]
        if i==j or _key(contacts[i])==_key(contacts[j]):continue
        when=beat/1000
        if a.note.type==NoteType.TAP:
            if min(contacts[i]['start'],contacts[j]['start'])<frozen_before-1e-9:continue
            # Keep the later Drags on their current physical finger; exchange
            # only the heads, then let the held contact collect its yellow tail.
            for lane in (None,.80,.86):
                value=list(contacts);valid=True
                for target,donor,old_id,new_id in ((i,j,a.id,b.id),(j,i,b.id,a.id)):
                    c=contacts[target];ids=[new_id if n==old_id else n for n in c['note_ids']]
                    if any(planner.original_tasks[n].note.type not in (NoteType.TAP,NoteType.DRAG) for n in ids):valid=False;break
                    anchor=point_at(contacts[donor]['points'],when)
                    if lane is not None:anchor=(anchor[0],lane)
                    new=_suffix(planner,c,ids,when,frozen_before,seed=anchor)
                    if new is None:valid=False;break
                    value[target]=new
                if valid:yield 'head_drag_handoff',value
        else:
            ids_by_index={k:[n for n in c['note_ids'] if n not in (a.id,b.id)] for k,c in enumerate(contacts)}
            targets=[]
            for task,key in ((a,_key(contacts[j])),(b,_key(contacts[i]))):
                candidates=[k for k,c in enumerate(contacts) if _key(c)==key and c['start']<when-.0011 and ids_by_index[k]]
                if not candidates:break
                target=max(candidates,key=lambda k:contacts[k]['start']);ids_by_index[target].append(task.id);targets.append(target)
            if len(targets)!=2:continue
            changed=set((i,j,*targets));value=[];valid=True
            for k,c in enumerate(contacts):
                if k not in changed:value.append(c);continue
                ids=ids_by_index[k]
                if not ids:
                    if c['start']<frozen_before-1e-9 or any(planner.original_tasks[n].note.type!=NoteType.DRAG for n in c['note_ids']):valid=False;break
                    continue
                new=_suffix(planner,c,ids,when,frozen_before)
                if new is None:valid=False;break
                value.append(new)
            if valid:yield 'simultaneous_drag_handoff',value


def proposals(planner,contacts,tasks,*,frozen_before,limit=PROPOSAL_LIMIT):
    """Finite geometric proposals; acceptance always needs a real rollout."""
    if not tasks:return
    from flowphi.planning.physical_plain import _touch_check
    from flowphi.planning.planner import State
    low=max(0.,min(t.start for t in tasks)/1000-.35)
    high=max(t.end/1000+.001 if t.note.type==NoteType.FLICK else t.beat/1000+.061 for t in tasks)
    streams=[_simultaneous(planner,contacts,tasks,frozen_before)]
    streams.extend(_flick_routes(planner,contacts,t,frozen_before) for t in tasks if t.note.type==NoteType.DRAG)
    seen=set();emitted=0;tested=0
    original_ids={id(c) for c in contacts}
    # Interleave motifs so a long Drag phrase cannot consume the entire budget.
    while streams and emitted<limit:
        remaining=[]
        for stream in streams:
            try:recipe,value=next(stream)
            except StopIteration:continue
            tested+=1
            if tested>GEOMETRY_LIMIT:return
            remaining.append(stream)
            value=sorted(value,key=lambda c:(c['start'],c['pointer']))
            signature=tuple((c['hand'],c['finger'],c['start'],c['end'],tuple(c['note_ids']),tuple(map(tuple,c['points']))) for c in value)
            if signature in seen:continue
            seen.add(signature)
            if not planner.contact_occupancy_valid(value) or _touch_check(planner,State(tuple(value)),tasks,(low,high)):continue
            valid=True
            for i,c in enumerate(value):
                if id(c) in original_ids:continue
                others=State(tuple(v for j,v in enumerate(value) if j!=i))
                errors,_,_=planner.constraints(c,others,-1 if c['hand']=='left' else 1,c['finger'],continuation=True)
                if errors:valid=False;break
            if not valid:continue
            emitted+=1
            yield recipe,value
            if emitted>=limit:break
        streams=remaining


def improve(planner,contacts,tasks,before,*,frozen_before,bounds=None,tick=None):
    """At most eight actual rollouts, across two composable motif rounds."""
    from flowphi.planning.physical_plain import _hard_rank,_interaction
    evaluator=planner._hand_evaluator
    best=list(contacts);report=before;history=[];attempted=0;checked=[]
    wanted={t.id for t in tasks}
    for turn in range(ROUNDS):
        winner=None;winner_report=report;recipe=None
        budget=min(ROLLOUT_LIMIT-attempted,ROLLOUT_LIMIT//ROUNDS)
        if budget<=0 or report['valid']:break
        for name,candidate in islice(proposals(planner,best,tasks,frozen_before=frozen_before,limit=budget),budget):
            attempted+=1
            motion=evaluator.evaluate(candidate,tasks,tick,time_range=bounds)
            checked.append(dict(recipe=name,hand_motion=motion,assignment=[
                (c['hand'],c['finger'],[n for n in c['note_ids'] if n in wanted])
                for c in candidate if wanted.intersection(c['note_ids'])]))
            rank=lambda m:(not m['valid'],_hard_rank(m) if not m['valid'] else (0,0.),_interaction(m),m['cost'])
            if rank(motion)<rank(winner_report) and _interaction(motion)<=_interaction(report)+.005:
                winner,winner_report,recipe=candidate,motion,name
        if winner is None:break
        best,report=winner,winner_report
        history.append(dict(recipe=recipe,valid=report['valid'],max_error_mm=report['max_error_mm']))
    return best,report,dict(attempted=attempted,rollout_limit=ROLLOUT_LIMIT,rounds=history,candidates=checked)
