"""Bounded physical selection of complete ordinary-note windows.

Coarse transfer/fatigue estimates remain diagnostics. They may propose a
candidate, but only the unchanged constrained skeleton can verify it.
"""
import time
import copy
from dataclasses import replace

WINDOW_SECONDS=1.2
SEARCH_WIDTH=8
ROLLOUT_LIMIT=8
KEEP_LIMIT=4


def _touch_check(planner,state,tasks,bounds):
    from flowphi.ui.model import PlanDocument
    from flowphi.ui.validation import quick_check,head_check
    from flowphi.ui.contact_state import clip_contact
    from flowphi.planning.contacts import validate_contact
    try:
        document=PlanDocument(dict(version=1,algorithm=5,timebase='chart_seconds',coordinates='normalized_screen',
            contacts=list(state.contacts),settings=vars(planner.settings),
            profile=planner.profile,physical_screen=list(planner.physical)),copy_baseline=False,allow_empty=True)
    except ValueError as exc:
        return [dict(code='invalid_controls',time=bounds[0],message=str(exc),
                     note_ids=[t.id for t in tasks])]
    notes=[(t.line,t.note) for _,t in sorted(planner.original_tasks.items())]
    covered={n for c in state.contacts for n in c['note_ids']}
    # A rounded future DOWN can start at this window's exclusive end while
    # its original beat falls a fraction of a millisecond before that end.
    # Match every owned/current head, without demanding unplanned future heads.
    expected=covered|{t.id for t in tasks}
    errors=quick_check(document,time_range=bounds)+head_check(document,notes,time_range=bounds,expected_ids=expected)
    for task in tasks:
        if task.id not in covered:
            errors.append(dict(code='missing_contact',time=task.beat/1000,note_ids=[task.id]))
    for c in state.contacts:
        if c['start']>=bounds[1] or c['end']<=bounds[0]:continue
        local=clip_contact(c,bounds)
        bad=validate_contact(planner._ribbon_geometry,local,local['points'],limit=1)
        if bad:errors.append(dict(code='geometry',time=bad[0],note_ids=c['note_ids']))
    return errors


def _signature(state,wanted):
    return tuple((c['hand'],c['finger'],tuple(n for n in c['note_ids'] if n in wanted))
                 for c in state.contacts if wanted.intersection(c['note_ids']))


def _trials(states,wanted):
    # First evaluate different complete assignments, then coordinate variants.
    groups={}
    for state in sorted(states,key=lambda s:s.score):
        groups.setdefault(_signature(state,wanted),[]).append(state)
    result=[]
    for depth in range(2):
        for group in groups.values():
            if depth<len(group):result.append(group[depth])
            if len(result)>=ROLLOUT_LIMIT:return result
    return result


def _hard_rank(motion):
    severity=(motion['max_error_mm']/4)**2+(motion['max_height_error_mm']/2)**2+(
        motion['skeleton_penetration_mm']/3)**2
    return len(motion.get('failed_note_ids',())),severity


def _interaction(motion):
    m=motion['interaction']
    return (m['projected_crossing_seconds']+2*m['palm_intrusion_seconds']+
            .05*m['tip_inversion_seconds'])


def run_ordinary(planner,console=None,*,tasks=None,initial_states=None,
                 return_states=False,on_unplanned=None,progress=None):
    from flowphi.planning.planner import State,PlanningError
    from flowphi.planning.scoring import score_history
    from flowphi.planning.hand_evaluation import HandEvaluator
    from flowphi.planning.junctions import ROLLOUT_LIMIT as junction_limit
    if not (planner.settings.hand_motion_repair and planner.settings.speed_limits):
        return planner._run_plain(console,tasks=tasks,initial_states=initial_states,
            return_states=return_states,on_unplanned=on_unplanned,progress=progress)
    tasks=planner.tasks if tasks is None else tasks
    if not tasks:
        states=list(initial_states or [State()])
        return states if return_states else list(min(states,key=lambda s:s.score).contacts)
    diagnostic=getattr(planner,'ordinary_hand_diagnostics',None)
    if diagnostic is None:
        diagnostic=planner.ordinary_hand_diagnostics=dict(windows=[],seconds=0.,
            window_seconds=WINDOW_SECONDS,search_width=SEARCH_WIDTH,
            rollout_limit=ROLLOUT_LIMIT,junction_rollout_limit=junction_limit,
            total_rollout_limit=ROLLOUT_LIMIT+junction_limit,keep_limit=KEEP_LIMIT)
    evaluator=getattr(planner,'_hand_evaluator',None)
    if evaluator is None:evaluator=planner._hand_evaluator=HandEvaluator(planner)
    started=time.monotonic()
    old_physical=getattr(planner,'_physical_plain_search',False)
    old_speculative=getattr(planner,'_speculative_transfers',False)
    old_window_start=getattr(planner,'_physical_window_start',None)
    planner._physical_plain_search=planner._speculative_transfers=True
    try:
        beam=[score_history(planner,s.contacts) for s in (initial_states or [State()])]
        cursor=0
        while cursor<len(tasks):
            stop=cursor+1
            while stop<len(tasks) and (tasks[stop].start-tasks[cursor].start<=WINDOW_SECONDS*1000 or
                                       tasks[stop].beat==tasks[stop-1].beat):stop+=1
            window=tasks[cursor:stop];wanted={t.id for t in window}
            begin=window[0].start/1000
            planner._physical_window_start=begin
            end=max(t.end/1000+.001 if t.note.type.name=='FLICK' else t.beat/1000+.061 for t in window)
            # Check sustained contacts throughout a gap, not just their heads.
            next_start=tasks[stop].start/1000 if stop<len(tasks) else float('inf')
            end=max(end,max((min(t.end/1000,next_start) for t in window),default=end))
            pending=[]
            search_started=time.monotonic()
            def tick(done,total,task=None):
                if progress:
                    progress(cursor,len(tasks),replace(window[0],planning_detail=dict(
                        start=begin,end=end,strategy='ordinary_hand',done=done,total=total,
                        elapsed_seconds=time.monotonic()-started)))
            states=planner._run_plain(console,tasks=window,initial_states=beam,return_states=True,
                on_unplanned=lambda t,r:pending.append((t,r)),width=SEARCH_WIDTH,progress=tick)
            # Reserve a small proposal budget for left/right spatial ownership.
            # It is an alternative, never a rule overriding occupancy or IK.
            spatial_pending=[]
            planner._physical_spatial_policy=True
            try:
                spatial=planner._run_plain(tasks=window,initial_states=beam,return_states=True,
                    on_unplanned=lambda t,r:spatial_pending.append(t.id),width=2,progress=tick)
            finally:
                planner._physical_spatial_policy=False
            if spatial_pending:spatial=[]
            search_seconds=time.monotonic()-search_started
            scene_ids={n for s in states for c in s.contacts for n in c['note_ids']}
            scene=[t for t in planner.tasks if begin-.35<=t.start/1000<end and t.id in scene_ids|wanted]
            bounds=(max(0.,begin-.35),end)
            evaluations=[]
            hand_seconds=touch_seconds=0.;structural_rejections=[]
            trials=_trials(spatial,wanted)[:2]+_trials(states,wanted)[:ROLLOUT_LIMIT-2]
            unique=[];seen=set()
            for state in trials:
                signature=tuple((c['hand'],c['finger'],c['start'],c['end'],tuple(c['note_ids']),
                    tuple(sorted(c.get('judgement_times',{}).items())),tuple(map(tuple,c['points']))) for c in state.contacts)
                if signature not in seen:unique.append(state);seen.add(signature)
            trials=unique
            for state in trials:
                check_started=time.monotonic()
                touch_errors=_touch_check(planner,state,window,bounds)
                touch_seconds+=time.monotonic()-check_started
                if touch_errors:
                    structural_rejections.extend(touch_errors)
                    continue
                check_started=time.monotonic()
                motion=evaluator.evaluate(state.contacts,scene,lambda a,b:tick(a,b),time_range=bounds)
                hand_seconds+=time.monotonic()-check_started
                evaluations.append((state,motion))
            base_evaluations=len(evaluations)
            junction_search=None;junction_seconds=0.
            if evaluations and not any(m['valid'] for _,m in evaluations):
                from flowphi.planning.junctions import improve
                seed,motion=min(evaluations,key=lambda item:(*_hard_rank(item[1]),item[0].score))
                junction_started=time.monotonic()
                revised,checked,junction_search=improve(planner,seed.contacts,scene,motion,
                    frozen_before=begin,bounds=bounds,tick=lambda a,b:tick(a,b))
                junction_seconds=time.monotonic()-junction_started
                if junction_search['rounds']:
                    evaluations.append((score_history(planner,revised),checked))
            feasible=[item for item in evaluations if item[1]['valid']]
            pool=feasible or evaluations
            if not pool:
                diagnostic['windows'].append(dict(start=begin,end=end,note_ids=sorted(wanted),
                    evaluated=0,valid=False,candidates=[],structural_rejections=structural_rejections,
                    timings=dict(search_seconds=search_seconds,hand_seconds=hand_seconds,touch_seconds=touch_seconds)))
                if on_unplanned is None:raise PlanningError('No structurally valid ordinary hand candidate')
                reasons={r['code']:1 for r in structural_rejections} or dict(no_complete_candidate=1)
                for t in window:on_unplanned(t,reasons)
                cursor=stop
                continue
            # Prefer uncrossed complete candidates when available. A necessary
            # crossing is still allowed if every physically valid option needs it.
            straight=[item for item in feasible if _interaction(item[1])<.01]
            if straight:pool=straight
            pool.sort(key=lambda item:((*_hard_rank(item[1]),) if not feasible else ())+
                      (item[0].score+item[1]['cost'],))
            selected=pool[:min(KEEP_LIMIT,planner.settings.beam_width)]
            report=dict(start=begin,end=end,note_ids=sorted(wanted),
                evaluated=len(evaluations),valid=bool(feasible),
                physical_evaluations=base_evaluations+(junction_search['attempted'] if junction_search else 0),
                junction_search=junction_search,
                timings=dict(search_seconds=search_seconds,hand_seconds=hand_seconds,touch_seconds=touch_seconds,junction_seconds=junction_seconds),
                structural_rejections=structural_rejections,
                candidates=[dict(score=s.score,hand_motion=m,assignment=list(_signature(s,wanted)))
                            for s,m in evaluations],selected_motion=selected[0][1])
            diagnostic['windows'].append(report)
            if not feasible and not planner.settings.allow_degraded:
                if on_unplanned is None:
                    raise PlanningError(f'No physically feasible ordinary window at {begin:.3f}s')
                reasons={reason:1 for _,m in evaluations for reason in m['reasons']}
                for t in window:on_unplanned(t,reasons)
                cursor=stop
                continue
            beam=[]
            for state,motion in selected:
                contacts=list(state.contacts)
                marker=next((i for i in range(len(contacts)-1,-1,-1)
                             if wanted.intersection(contacts[i]['note_ids'])),None)
                if marker is not None:
                    c=contacts[marker]
                    contacts[marker]=dict(c,ordinary_decisions=[*c.get('ordinary_decisions',()),
                        dict(window=window[0].id,start=begin,end=end,hand_motion=motion)])
                beam.append(State(tuple(contacts),state.score+motion['cost']))
            covered={n for s in beam for c in s.contacts for n in c['note_ids']}
            for t,reasons in pending:
                if t.id in covered:continue
                if on_unplanned:on_unplanned(t,reasons)
                else:raise PlanningError(f'No feasible ordinary contact for note {t.id}')
            if console:console.print(f'ORDINARY_HAND {begin:.3f}–{end:.3f}s '
                f'{len(feasible)}/{len(evaluations)} physically valid')
            cursor=stop
    finally:
        planner._physical_plain_search=old_physical
        planner._speculative_transfers=old_speculative
        planner._physical_window_start=old_window_start
        diagnostic['seconds']+=time.monotonic()-started
    # Restore all original heuristic warnings/costs. Physical validity is a
    # separate proof with an explicit scope, never deletion of failed checks.
    beam=[score_history(planner,s.contacts) for s in beam]
    return beam if return_states else list(min(beam,key=lambda s:s.score).contacts)


def audit_ordinary(planner,contacts,progress=None):
    """Refresh proofs after future preparation, joining and final repairs."""
    from flowphi.planning.planner import State
    diagnostic=getattr(planner,'ordinary_hand_diagnostics',None)
    if not diagnostic:return
    started=time.monotonic();evaluator=planner._hand_evaluator
    for report in diagnostic['windows']:
        low=max(0.,report['start']-.35);high=report['end']
        scene=[t for t in planner.tasks if low<=t.start/1000<high]
        def tick(done,total):
            if progress:progress(len(planner.tasks),len(planner.tasks),replace(scene[0],
                planning_detail=dict(start=report['start'],end=high,strategy='ordinary_audit',
                                     done=done,total=total,elapsed_seconds=time.monotonic()-started)))
        motion=evaluator.evaluate(contacts,scene,tick,time_range=(low,high))
        errors=_touch_check(planner,State(tuple(contacts)),scene,(low,high))
        if errors:
            motion=copy.deepcopy(motion);motion['valid']=False
            motion['reasons']=sorted(set(motion['reasons'])|{e['code'] for e in errors})
            motion['touch_errors']=errors
        report['final_motion']=motion
    diagnostic['final_failed_windows']=sum(not w['final_motion']['valid'] for w in diagnostic['windows'])
    diagnostic['audit_seconds']=time.monotonic()-started
