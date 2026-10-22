"""Bounded full-context repairs; large transfers require actual hard verification.

Proposals are not rejected by the coarse transfer estimate or movement price.
Occupancy, independent DOWNs, millisecond geometry and the constrained skeleton
remain authoritative. Previously feasible windows must stay feasible.
"""
import copy
import math
import time
from collections import Counter

from flowphi.motion import FINGER_ORDER, point_at, world_xy
from flowphi.charts.model import NoteType
from flowphi.planning.contacts import project, validate_contact, refine_contact

MAX_PROPOSALS = 23


def _occupancy_conflicts(contacts):
    active,conflicts={},set()
    for i,c in sorted(enumerate(contacts),key=lambda row:row[1]['start']):
        key=c['hand'],c['finger']
        live=[(j,old) for j,old in active.get(key,()) if old['end']>c['start']+1e-9]
        for j,old in live:
            conflicts.add((min(i,j),max(i,j),key,max(c['start'],old['start']),min(c['end'],old['end'])))
        active[key]=[*live,(i,c)]
    return conflicts


def _severity(report):
    return (report['max_error_mm']/4.)**2+(report['max_height_error_mm']/2.)**2+(
        report['skeleton_penetration_mm']/3.)**2+20.*('actual_flick_motion' in report['reasons'])


def _rank(report):
    interaction = report.get('interaction', {})
    return (not report['valid'], _severity(report) if not report['valid'] else 0.,
            interaction.get('projected_crossing_seconds', 0.)+
            interaction.get('palm_intrusion_seconds', 0.), report['cost'])


def _crossing(report):
    values=report.get('interaction',{})
    return (values.get('projected_crossing_seconds',0.)+2*values.get('palm_intrusion_seconds',0.)+
            .05*values.get('tip_inversion_seconds',0.))


def _placements(planner, contacts, tasks, report):
    from flowphi.planning.ribbons import _geometry
    geometry = _geometry(planner)
    rejects=planner._repair_rejections=Counter()
    planner._repair_bad=[]
    low = min(t.start for t in tasks)/1000
    high = max(t.end/1000 if t.note.type == NoteType.FLICK else t.beat/1000+.013 for t in tasks)
    wanted = {t.id for t in tasks}
    cut = low-.30
    selected = [i for i,c in enumerate(contacts) if c.get('planned', True) and
                (wanted.intersection(c['note_ids']) or cut <= c['start'] <= high+.15)]
    known_conflicts=_occupancy_conflicts(contacts)
    # Changing whole merged records cannot express a yellow handoff while
    # keeping each blue head. Try those bounded note-level junctions first.
    from flowphi.planning.junctions import proposals
    junction_count=0
    originals={id(c) for c in contacts}
    for name,candidate in proposals(planner,contacts,tasks,frozen_before=cut,limit=8):
        changed=[i for i,c in enumerate(candidate) if id(c) not in originals]
        if not changed or not _occupancy_conflicts(candidate)<=known_conflicts:continue
        junction_count+=1
        yield name,candidate,changed
    # Move whole phrases to coherent lanes before considering local overrides.
    recipes = [('offset_all',(0.,-.025)),('offset_all',(0.,.025)),
               ('lanes', (.65,.85)), ('lanes', (.85,.65)), ('lanes', (.45,.85)),
               ('lanes', (.85,.45)), ('lanes', (.65,.65)), ('lanes', (.85,.85)),
               ('swap', None)]
    worst = report.get('worst', [])
    hand = worst[0]['hand'] if worst else 'left'
    recipes.extend((hand, (y,y)) for y in (.35,.60,.85))
    recipes.extend((other, (y,y)) for other,y in (('right' if hand=='left' else 'left',.45),
                                                ('right' if hand=='left' else 'left',.85)))
    worst_ids=set(worst[0]['note_ids']) if worst else wanted
    # A bad landing can be caused by the previous phrase choosing the wrong
    # hand. Swap the lead-in, stopping before the failed contact itself; swapping
    # the entire window would put the landing on the wrong hand again.
    failed=next((c for c in contacts if worst_ids.intersection(c['note_ids'])),None)
    if failed:
        stop=failed['start']
        starts=sorted({contacts[i]['start'] for i in selected
            if stop-.8<=contacts[i]['start']<stop and
            ((contacts[i]['hand']=='left' and contacts[i]['points'][0][1]>.53) or
             (contacts[i]['hand']=='right' and contacts[i]['points'][0][1]<.47))})[:3]
        recipes[2:2]=[('swap_lead',(start,stop)) for start in starts]
    if 'actual_flick_motion' in report['reasons']:
        recipes.extend(('flick',angle) for angle in (math.pi,math.pi/2,-math.pi/2))
    recipes.extend(('offset',delta) for delta in ((0.,-.025),(0.,-.06),(0.,.025),(0.,.06),(-.04,0.),(.04,0.)))
    for name, lanes in recipes[:MAX_PROPOSALS-junction_count]:
        candidate = list(contacts)
        changed, failed = [], False
        for i in selected:
            old = contacts[i]
            if name not in ('lanes','swap','swap_lead','offset','offset_all','flick',old['hand']):
                continue
            if name=='offset' and not worst_ids.intersection(old['note_ids']):
                continue
            if name=='flick' and (old['kind']!='flick' or old['start']<cut):
                continue
            new = copy.deepcopy(old)
            if name in ('swap','swap_lead'):
                if name=='swap_lead' and not lanes[0]<=old['start']<lanes[1]:
                    continue
                if old['start'] < cut:
                    continue
                new['hand'] = 'right' if old['hand']=='left' else 'left'
                new['pointer'] = (0 if new['hand']=='left' else 5)+FINGER_ORDER.index(new['finger'])
            elif name=='flick':
                anchor=point_at(old['points'],old.get('beat',old['start']))
                cosine,sine=math.cos(lanes),math.sin(lanes)
                new['points']=[[t,anchor[0]+(x-anchor[0])*cosine-(y-anchor[1])*sine,
                    anchor[1]+(x-anchor[0])*sine+(y-anchor[1])*cosine] for t,x,y in old['points']]
            else:
                begin = max(old['start'],cut)
                anchor = max(begin,min(old['end']-.001,old.get('beat',old['start'])))
                origin = point_at(old['points'], anchor)
                preferred = (tuple(a+b for a,b in zip(origin,lanes)) if name in ('offset','offset_all') else
                    (.28 if old['hand']=='left' else .72, lanes[old['hand']=='right']))
                feasible = project(geometry.zone(new,anchor),preferred)
                if feasible is None:
                    rejects['empty_anchor']+=1
                    failed=True
                    break
                delta = tuple(a-b for a,b in zip(feasible,origin))
                times = sorted({p[0] for p in old['points']} | {begin} |
                    {float(t) for t in geometry.times(old,.02) if t>=begin} |
                    {min(old['end']-.001,begin+.10), min(old['end']-.001,begin+.20)})
                points = []
                for t in times:
                    if not old['start'] <= t <= old['end']-.001+1e-10:
                        continue
                    xy = point_at(old['points'],t)
                    if t >= begin:
                        # Preserve a frozen prefix and ramp a changed suffix.
                        u = min(1.,max(0.,(t-begin)/.20)) if begin>old['start'] else 1.
                        target = tuple(a+u*b for a,b in zip(xy,delta))
                        xy = project(geometry.zone(new,t),target)
                        if xy is None:
                            rejects['empty_zone']+=1
                            failed=True
                            break
                    points.append([float(t),float(xy[0]),float(xy[1])])
                if failed:
                    break
                new['points'] = points
            if new['hand']==old['hand'] and new['points']==old['points']:
                continue
            bad=validate_contact(geometry,new,new['points'],limit=1)
            if bad and name!='swap':
                frozen=max(new['start'],cut)
                suffix=dict(new,start=frozen,points=[[frozen,*point_at(new['points'],frozen)],
                    *[p for p in new['points'] if p[0]>frozen]])
                if frozen>new['start']:
                    suffix['refinement_start']=list(point_at(old['points'],frozen))
                    suffix['points'][0]=[frozen,*suffix['refinement_start']]
                extra=set(bad)
                for _ in range(2):
                    trial,_=refine_contact(geometry,suffix,iterations=16,extra_times=extra)
                    if trial is None:
                        break
                    new['points']=[*[p for p in old['points'] if p[0]<frozen],*trial]
                    bad=validate_contact(geometry,new,new['points'],limit=4)
                    if not bad:
                        break
                    extra.update(bad)
            if bad:
                rejects['geometry']+=1
                if len(planner._repair_bad)<12:
                    planner._repair_bad.append(dict(recipe=name,ids=new['note_ids'],kind=new['kind'],at=bad[0],
                        before=point_at(old['points'],bad[0]),after=point_at(new['points'],bad[0])))
                failed=True
                break
            if any(math.dist(world_xy(a[1:],planner.physical),world_xy(b[1:],planner.physical)) >
                   planner.settings.finger_speed*(b[0]-a[0])+1e-9
                   for a,b in zip(new['points'],new['points'][1:])):
                rejects['contact_speed']+=1
                failed=True
                break
            candidate[i] = new
            changed.append(i)
        if failed or not changed or not _occupancy_conflicts(candidate) <= known_conflicts:
            if not failed:
                rejects['occupancy_or_unchanged']+=1
            continue
        from flowphi.planning.ribbons import _heads_ok
        if not _heads_ok(planner,candidate,planner.tasks):
            rejects['heads']+=1
            continue
        yield name+(':'+str(lanes) if lanes is not None else ''), candidate, changed


def repair_failed_windows(planner, contacts, progress=None, *, windows=None):
    """Try at most 23 placements/window with unchanged exact hard checks."""
    evaluator = planner._hand_evaluator
    windows = planner.ribbon_diagnostics['windows'] if windows is None else windows
    all_windows = planner.ribbon_diagnostics['windows']
    started = time.monotonic()
    results = []
    transactions=[]
    initial={w['id']:w['hand_motion'] for w in all_windows}
    protected={key for key,m in initial.items() if m['valid']}
    initial_severity=sum(_severity(m) for m in initial.values() if not m['valid'])
    initial_crossing=sum(_crossing(m) for m in initial.values())
    lookup={t.id:t for t in getattr(planner,'tasks',())}
    for window in windows:
        baseline = window['hand_motion']
        if baseline['valid']:
            continue
        tasks=sorted((lookup[n] for n in window.get('note_ids',())),key=lambda t:(t.start,t.id))
        if transactions:
            baseline=evaluator.evaluate(contacts,tasks)
            window.update(hand_motion=baseline,hand_motion_degraded=not baseline['valid'])
            if baseline['valid']:
                protected.add(window['id'])
                continue
        if 'contact_state' in baseline['reasons']:
            results.append(dict(start=window['start'],end=window['end'],attempted=0,accepted=False,
                placement=None,before=baseline,after=baseline,changed_contacts=[],
                rejections=dict(unresolved_contact_lifecycle=1)))
            continue
        best, best_report, label, changes = None, baseline, None, []
        attempted = 0
        def tick(done,total):
            if progress:
                from dataclasses import replace
                progress(len(planner.tasks),len(planner.tasks),replace(tasks[0],planning_detail=dict(
                    start=window['start'],end=window['end'],strategy='hand_repair',done=done,total=total,
                    elapsed_seconds=time.monotonic()-started)))
        for name,candidate,changed in _placements(planner,contacts,tasks,baseline):
            attempted += 1
            report = evaluator.evaluate(candidate,tasks,tick)
            if _crossing(report)>_crossing(baseline)+.005:
                continue
            if _rank(report) < _rank(best_report):
                # Never introduce a new hard failure type to improve a cost.
                if not report['valid'] and not set(report['reasons']) <= set(baseline['reasons']):
                    continue
                best,best_report,label,changes = candidate,report,name,changed
                interaction=report.get('interaction',{})
                if report['valid'] and interaction.get('projected_crossing_seconds',0.)<1e-6 and interaction.get('palm_intrusion_seconds',0.)<1e-6:
                    break
        accepted = False
        if best is not None:
            # Verify local neighbours now. Batch the expensive whole-history
            # audit below, with reverse rollback if any protected window fails.
            low=max(-1.,window['start']-.55)
            high=max(best[i]['end'] for i in changes)+.25
            context=[t for t in planner.tasks if t.start/1000<high and t.end/1000>=low]
            before_context=evaluator.evaluate(contacts,context,tick,time_range=(low,high))
            after_context=evaluator.evaluate(best,context,tick,time_range=(low,high))
            safe=after_context['valid'] or (
                set(after_context['reasons'])<=set(before_context['reasons']) and
                set(after_context['failed_note_ids'])<=set(before_context['failed_note_ids']) and
                _severity(after_context)<=_severity(before_context)+1e-8)
            safe=safe and _crossing(after_context)<=_crossing(before_context)+.005
            if safe:
                transactions.append((list(contacts),{w['id']:w['hand_motion'] for w in all_windows},
                                     set(protected),len(results)))
                for i in changes:
                    c=best[i]
                    c['decision']=dict(c.get('decision',{}),hand_repair=dict(window=window['id'],
                        placement=label,hand_valid=best_report['valid'],coarse_transfer_replaced_by_rollout=True))
                contacts[:]=best
                window.update(hand_motion=best_report,hand_motion_degraded=not best_report['valid'])
                if best_report['valid']:
                    protected.add(window['id'])
                accepted=True
        results.append(dict(start=window['start'],end=window['end'],attempted=attempted,
            accepted=accepted,placement=label if accepted else None,
            before=baseline,after=window['hand_motion'],changed_contacts=changes if accepted else [],
            rejections=dict(getattr(planner,'_repair_rejections',{}))))
        results[-1]['geometry_samples']=getattr(planner,'_repair_bad',[])
    rollbacks=0
    while transactions:
        audits={}
        for other in all_windows:
            scene=sorted((lookup[n] for n in other['note_ids']),key=lambda t:(t.start,t.id))
            audits[other['id']]=evaluator.evaluate(contacts,scene,tick)
        if (all(audits[key]['valid'] for key in protected) and
                sum(_severity(m) for m in audits.values() if not m['valid'])<=initial_severity+1e-8 and
                sum(_crossing(m) for m in audits.values())<=initial_crossing+.005):
            for other in all_windows:
                motion=audits[other['id']]
                other.update(hand_motion=motion,hand_motion_degraded=not motion['valid'])
            break
        old_contacts,old_reports,protected,index=transactions.pop()
        contacts[:]=old_contacts
        for other in all_windows:
            motion=old_reports[other['id']]
            other.update(hand_motion=motion,hand_motion_degraded=not motion['valid'])
        results[index].update(accepted=False,placement=None,changed_contacts=[],rolled_back=True)
        rollbacks+=1
    planner.ribbon_diagnostics['hand_motion_failed_windows']=sum(not w['hand_motion']['valid'] for w in all_windows)
    final={(w['start'],w['end']):w['hand_motion'] for w in all_windows}
    for result in results:
        result['after']=final[result['start'],result['end']]
    summary=dict(seconds=time.monotonic()-started,proposal_limit=MAX_PROPOSALS,windows=results,
        rollbacks=rollbacks,
        accepted=sum(r['accepted'] for r in results),remaining=planner.ribbon_diagnostics['hand_motion_failed_windows'])
    planner.ribbon_diagnostics['hand_repair']=summary
    return summary
