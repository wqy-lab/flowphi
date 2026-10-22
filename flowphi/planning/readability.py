"""Bounded counterfactual phrase search with unchanged final hard checks."""
import copy
import math
import time
from dataclasses import replace

MAX_WINDOWS=2
MAX_SECONDS=12.
MAX_ROLLOUTS=4
HANDOFF_ALLOWANCE_M=.012
HANDOFF_MAX_GAP_SECONDS=.65


def handoff_transfers(planner,contacts,low,high):
    """Neutral wrist travel across each immutable/mutable boundary.

    A gap between DOWNs is not a safe handoff by itself: the following contact
    can still belong to a different normal lane. This is a soft repair guard,
    independent of the unchanged realised-skeleton speed and reach checks.
    """
    result={}
    for hand in ('left','right'):
        rows=sorted((c for c in contacts if c['hand']==hand),key=lambda c:c['start'])
        for edge,boundary in (('entry',low),('exit',high)):
            following=next((c for c in rows if c['start']>=boundary),None)
            if following is None or (edge=='entry' and following['start']>=high):continue
            previous=max((c for c in rows if c['start']<boundary),
                         key=lambda c:c['end'],default=None)
            if previous is None:continue
            gap=following['start']-previous['end']
            # Longer pauses have a separate withdrawal/return planner. Keep
            # this guard on rapid phrase-to-phrase transitions, not rests.
            if not 0<=gap<=HANDOFF_MAX_GAP_SECONDS:continue
            a=planner.anchor(previous,previous['end'])
            b=planner.anchor(following,following['start'])
            result[edge,hand]=dict(distance_m=math.dist(a,b),gap_seconds=gap,
                from_note_ids=previous['note_ids'],to_note_ids=following['note_ids'])
    return result


def handoff_regressions(planner,before,after,low,high):
    original=handoff_transfers(planner,before,low,high)
    proposed=handoff_transfers(planner,after,low,high)
    return [dict(edge=key[0],hand=key[1],before=original[key],after=value)
            for key,value in proposed.items() if key in original and
            value['distance_m']>original[key]['distance_m']+HANDOFF_ALLOWANCE_M]


def acceptable(before,after):
    if after['valid']:
        safe=True
    else:
        def severity(m):
            return (m['max_error_mm']/4)**2+(m['max_height_error_mm']/2)**2+(m['skeleton_penetration_mm']/3)**2
        safe=(set(after['reasons'])<=set(before['reasons']) and
              set(after['failed_note_ids'])<=set(before['failed_note_ids']) and
              severity(after)<=severity(before)+1e-8)
    for key in ('projected_crossing_seconds','palm_intrusion_seconds'):
        safe=safe and after['interaction'][key]<=before['interaction'][key]+.005
    return safe and after['occlusion']['weighted_note_seconds']<before['occlusion']['weighted_note_seconds']-.01


def repair_readability(planner,contacts,progress=None):
    from flowphi.planning.scoring import line_distance_cost,score_history
    from flowphi.planning.planner import Planner,State
    from flowphi.planning.hand_evaluation import HandEvaluator
    from flowphi.ui.model import PlanDocument
    from flowphi.ui.validation import validate
    started=time.monotonic();bad=[]
    for c in contacts:
        crossed=c.get('decision',{}).get('interaction',{}).get('peak_inversion',0.)>=.75
        high=planner.settings.judge_line_weight and line_distance_cost(planner,c)>.1
        if not high and not (crossed and planner.settings.hand_interaction_weight):
            continue
        bad.append(c)
    from flowphi.planning.occlusion import scorer
    visibility=scorer(planner)
    weighted=[(c,visibility.contact(c)) for c in contacts]
    ribbons=getattr(planner,'ribbon_diagnostics',{}).get('windows',())
    ranked=[]
    for center in {round(c['start']) for c in bad}:
        begin,end=max(0.,center-2.5),center+2.5
        # Those phrases already have their own whole-window geometry and hand
        # search. Keep this small extra budget for ordinary note sequences.
        if any(begin<w['end'] and end>w['start'] for w in ribbons):continue
        if sum(c['start']<end and c['end']>begin for c in bad)<3:continue
        priority=sum(weight*max(0.,min(end,c['end'])-max(begin,c['start']))/
                     max(.001,c['end']-c['start']) for c,weight in weighted)
        ranked.append((begin,end,priority))
    episodes=[]
    for e in sorted(ranked,key=lambda e:e[2],reverse=True):
        if not any(e[0]<old[1] and e[1]>old[0] for old in episodes):
            episodes.append(e)
            if len(episodes)>=MAX_WINDOWS:break
    reports=[]
    for begin,end,_ in episodes:
        high=end+.3
        until=-1.
        for c in contacts:
            if c['start']>=high and c['start']-until>.04:
                high=(until+c['start'])/2;break
            until=max(until,c['end'])
        low=max(0.,high-MAX_SECONDS)
        low=math.ceil(low*2)/2
        # Use a preceding free gap where possible, without enlarging the budget.
        gaps=[];until=-1.
        for c in contacts:
            if low<=c['start']<=low+.5 and c['start']-until>.04:
                gaps.append((until+c['start'])/2)
            until=max(until,c['end'])
        if gaps:low=max(low,gaps[0])
        if any(low<r['end'] and high>r['start'] for r in reports):continue
        options=replace(planner.settings,beam_width=min(8,planner.settings.beam_width),
                        candidates_per_finger=min(3,planner.settings.candidates_per_finger))
        local=Planner(planner.chart,options,planner.profile)
        tasks=[t for t in local.tasks if low<=t.start/1000<high]
        if not tasks:continue
        local._speculative_transfers=True
        def tick(done,total,task=None):
            if progress:
                progress(len(planner.tasks),len(planner.tasks),replace(tasks[0],planning_detail=dict(
                    start=low,end=high,strategy='readability',done=done,total=total,
                    elapsed_seconds=time.monotonic()-started)))
        prefix=tuple(c for c in contacts if c['start']<low)
        initial=State(prefix,score_history(local,prefix).score) if prefix else State()
        pending=[]
        states=local._run_plain(tasks=tasks,initial_states=[initial],return_states=True,
            on_unplanned=lambda t,r:pending.append(t.id),progress=tick)
        report=dict(start=low,end=high,pending=pending,accepted=False,evaluated=0,
                    handoff_rejections=[])
        reports.append(report)
        if pending:continue
        local._speculative_transfers=False
        evaluator=HandEvaluator(local)
        scene=[t for t in local.tasks if low-.3<=t.start/1000<high+.3]
        bounds=(max(0.,low-.3),high+.3)
        before=evaluator.evaluate(contacts,scene,lambda a,b:tick(a,b),time_range=bounds)
        report['before']=before
        coordinate_attempted=False
        def signature(c):
            return (c['hand'],c['finger'],c['start'],c['end'],tuple(c['note_ids']),tuple(map(tuple,c['points'])))
        original={signature(c):c for c in contacts}
        wanted={t.id for t in tasks}
        for state in sorted(states,key=lambda s:s.score)[:MAX_ROLLOUTS]:
            candidate=sorted([*[original.get(signature(c),c) for c in state.contacts],
                              *[c for c in contacts if c['start']>=high]],key=lambda c:(c['start'],c['pointer']))
            handoff=handoff_regressions(local,contacts,candidate,low,high)
            if handoff:
                report['handoff_rejections'].append(dict(stage='proposal',transfers=handoff))
                continue
            baseline=dict(version=1,algorithm=5,timebase='chart_seconds',coordinates='normalized_screen',
                contacts=candidate,settings=vars(planner.settings),profile=planner.profile,physical_screen=list(planner.physical))
            native=validate(PlanDocument(baseline),planner.chart,time_range=bounds)
            if not native['valid']:continue
            after=evaluator.evaluate(candidate,scene,lambda a,b:tick(a,b),time_range=bounds)
            if not after['valid'] and not coordinate_attempted and after.get('worst'):
                coordinate_attempted=True
                at=after['worst'][0]['time']
                focus=[t for t in local.tasks if at-.5<=t.start/1000<at+.3]
                if focus:
                    from flowphi.planning.repair import repair_failed_windows
                    local._hand_evaluator=evaluator
                    motion=evaluator.evaluate(candidate,focus)
                    window=dict(id=-1,start=focus[0].start/1000,end=focus[-1].start/1000,
                        note_ids=[t.id for t in focus],hand_motion=motion)
                    local.ribbon_diagnostics=dict(windows=[window])
                    report['coordinate_repair']=repair_failed_windows(local,candidate,
                        lambda a,b,t:tick(a,b,t))
                    after=evaluator.evaluate(candidate,scene,lambda a,b:tick(a,b),time_range=bounds)
            report['evaluated']+=1
            report['after']=after
            # Coordinate repairs can also change the first/last active hand.
            handoff=handoff_regressions(local,contacts,candidate,low,high)
            if handoff:
                report['handoff_rejections'].append(dict(stage='coordinate_repair',transfers=handoff))
                continue
            if not acceptable(before,after) or (not after['valid'] and not planner.settings.allow_degraded):
                continue
            # A complete successful rollout alone can clear transfer estimates.
            # A partial improvement remains an explicitly failed draft.
            for i,c in enumerate(candidate):
                if not wanted.intersection(c['note_ids']):continue
                c=candidate[i]=copy.deepcopy(c)
                decision=c.setdefault('decision',{})
                transfer=decision.pop('pending_transfer_verification',[])
                decision['readability_rollout']=dict(valid=after['valid'],start=low,end=high)
                if after['valid'] and transfer:
                    decision['degraded']=[v for v in decision['degraded'] if v not in transfer]
                elif transfer:
                    decision.setdefault('costs',{})['degraded']=1000.*len(decision['degraded'])
            contacts[:]=candidate;report['accepted']=True
            break
    planner.readability_diagnostics=dict(seconds=time.monotonic()-started,windows=reports,
        window_limit=MAX_WINDOWS,rollout_limit=MAX_ROLLOUTS,window_seconds_limit=MAX_SECONDS,
        handoff=dict(allowance_m=HANDOFF_ALLOWANCE_M,max_gap_seconds=HANDOFF_MAX_GAP_SECONDS,
                     model='neutral_wrist_anchor'))
    return planner.readability_diagnostics
