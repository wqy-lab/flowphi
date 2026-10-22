import unittest
from types import SimpleNamespace

from flowphi.charts.model import NoteType
from flowphi.planning.planner import State
from flowphi.planning.chords import chord_group, chord_states, ordered, MAX_ROLLOUTS
from flowphi.planning.planner import Planner, Settings
from flowphi.motion import default_profile
from test_drag_continuation import make_chart


def task(n, x):
    return SimpleNamespace(id=n, start=1000, end=1060, x=x, note=SimpleNamespace(type=NoteType.TAP))


class ChordTests(unittest.TestCase):
    def test_flick_preparation_and_tap_are_grouped_by_judgement_beat(self):
        p=Planner(make_chart([(4,1.,-.7,0.),(1,1.,.7,0.)]),
                  Settings(fingers=('index',)),default_profile())
        group=chord_group(p.tasks,0,('index',))
        self.assertEqual(2,len(group))
        self.assertNotEqual(group[0].start,group[1].start)
        contacts=p._run_plain()
        self.assertEqual({'left'},{c['hand'] for c in contacts if c['kind']=='flick'})
        self.assertTrue(ordered(contacts,{0,1},p.physical))

    def test_flick_hold_tap_keeps_two_independent_press_heads_and_swipe(self):
        from flowphi.ui.model import PlanDocument
        from flowphi.ui.validation import head_check
        chart=make_chart([(4,1.,2.,0.),(3,1.,2.,1.),(1,1.,-2.,0.)])
        p=Planner(chart,Settings(fingers=('index',)),default_profile())
        self.assertEqual(3,len(chord_group(p.tasks,0,('index',))))
        contacts=p._run_plain()
        self.assertEqual({0,1,2},{n for c in contacts for n in c['note_ids']})
        self.assertEqual(2,len(contacts))
        merged=next(c for c in contacts if 0 in c['note_ids'])
        self.assertEqual({0,1},set(merged['note_ids']))
        self.assertEqual('right',merged['hand'])
        self.assertGreater(max(abs(pt[2]-merged['points'][0][2]) for pt in merged['points']),.1)
        doc=PlanDocument(dict(version=1,algorithm=5,timebase='chart_seconds',coordinates='normalized_screen',
            contacts=contacts,profile=p.profile,settings=vars(p.settings),physical_screen=list(p.physical)))
        notes=[(line,note) for line in chart.lines for note in line.notes]
        self.assertEqual([],head_check(doc,notes))

    def test_real_planner_keeps_both_independent_heads_and_spatial_hands(self):
        p=Planner(make_chart([(1,1.,-.7,0.),(1,1.,.7,0.)]),
                  Settings(fingers=('index',)),default_profile())
        contacts=p._run_plain()
        self.assertEqual({0,1},{n for c in contacts for n in c['note_ids']})
        self.assertEqual(2,len(contacts))
        self.assertEqual({'left','right'},{c['hand'] for c in contacts})
        self.assertTrue(ordered(contacts,{0,1},p.physical))

    def planner(self, *, transfers=False, valid=False):
        calls=[]
        def choices(t, state, degraded=False):
            occupied={c['hand'] for c in state.contacts if c['start']==1.}
            result=[]
            for h in ('left','right'):
                if h in occupied:
                    continue
                spatial=(h=='left')==(t.x<.5)
                if transfers and spatial and not degraded:
                    continue
                errors=['wrist_transfer'] if transfers and spatial else []
                c=dict(hand=h,finger='index',pointer=0 if h=='left' else 5,kind='tap',
                    start=1.,end=1.061,note_ids=[t.id],points=[[1.,t.x,.8]],
                    decision=dict(costs=dict(degraded=1000.*len(errors)),degraded=errors))
                result.append((10. if spatial else 0.,c))
            return result
        def evaluate(*args, **kwargs):
            calls.append(kwargs)
            return dict(valid=valid)
        p=SimpleNamespace(choices=choices,physical=(.28,.1575),settings=SimpleNamespace(beam_width=8,allow_degraded=True),
            _hand_evaluator=SimpleNamespace(evaluate=evaluate),tasks=[])
        return p,calls

    def test_failed_whole_chord_stays_a_failed_draft_and_never_clears_caps(self):
        p,calls=self.planner(transfers=True,valid=False)
        original=p.choices
        def choices(t,state,degraded=False):
            if not degraded:
                return []
            result=original(t,state,True)
            for _,c in result:
                c['decision']['degraded']=['wrist_transfer']
                c['decision']['costs']['degraded']=1000.
            return result
        p.choices=choices
        group=[task(0,.3),task(1,.7)]
        states=chord_states(p,group,[State()])
        self.assertTrue(ordered(states[0].contacts,{0,1},p.physical))
        self.assertTrue(all(c['decision']['degraded'] for c in states[0].contacts))
        self.assertTrue(all(not c['decision']['spatial_chord_draft']['valid'] for c in states[0].contacts))
        self.assertTrue(all('verified_transfer' not in c['decision'] for c in states[0].contacts))
        self.assertLessEqual(len(calls),MAX_ROLLOUTS)
        p.settings.allow_degraded=False
        self.assertEqual([],chord_states(p,group,[State()]))

    def test_joint_assignment_prefers_legal_order_over_cheaper_crossing(self):
        p,_=self.planner();group=[task(1,.3),task(0,.7)]
        states=chord_states(p,group,[State()])
        self.assertTrue(states)
        self.assertTrue(all(ordered(s.contacts,{0,1},p.physical) for s in states))
        # IDs and visitation order must not define the spatial ordering.
        reverse=chord_states(p,group[::-1],[State()])
        self.assertEqual({(c['hand'],c['note_ids'][0]) for c in states[0].contacts},
                         {(c['hand'],c['note_ids'][0]) for c in reverse[0].contacts})

    def test_coarse_exception_requires_hard_rollout(self):
        p,calls=self.planner(transfers=True,valid=False);group=[task(0,.3),task(1,.7)]
        states=chord_states(p,group,[State()])
        self.assertTrue(calls)
        self.assertLessEqual(len(calls),MAX_ROLLOUTS)
        self.assertFalse(ordered(states[0].contacts,{0,1},p.physical))
        p,calls=self.planner(transfers=True,valid=True)
        states=chord_states(p,group,[State()])
        self.assertTrue(ordered(states[0].contacts,{0,1},p.physical))
        self.assertTrue(all(c['decision']['verified_transfer']['valid'] for c in states[0].contacts))
        self.assertTrue(all(not c['decision']['degraded'] for c in states[0].contacts))

    def test_motion_inside_chord_can_invert_after_down(self):
        pair=[dict(hand='left',note_ids=[0],start=1.,end=1.1,points=[[1.,.3,.8],[1.05,.8,.8]]),
              dict(hand='right',note_ids=[1],start=1.,end=1.1,points=[[1.,.7,.8]])]
        self.assertFalse(ordered(pair,{0,1},(.28,.1575)))

    def test_group_is_exactly_two_simultaneous_heads(self):
        group=[task(0,.3),task(1,.7)]
        self.assertEqual(group,chord_group(group,0,('index',)))
        self.assertIsNone(chord_group(group,1,('index',)))
        self.assertIsNone(chord_group(group+[task(2,.5)],0,('index',)))
        self.assertIsNone(chord_group(group,0,('index','middle')))
        group[1].note.type=NoteType.DRAG
        self.assertEqual(group,chord_group(group,0,('index',)))


if __name__=='__main__':
    unittest.main()
