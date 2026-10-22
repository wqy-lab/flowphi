from types import SimpleNamespace
import unittest

from flowphi.charts.model import NoteType, NoteState
from flowphi.motion import default_profile
from flowphi.planning.occlusion import OcclusionScorer, MAX_CACHE, _remember, _hull, _inside
from flowphi.planning.planner import Planner, Settings
from flowphi.planning.scoring import score_contact
from test_drag_continuation import make_chart


class OcclusionTests(unittest.TestCase):
    def test_fast_estimate_includes_idle_middle_finger(self):
        p,visibility=self.fixture()
        c=dict(hand='left',finger='index',start=1.,end=1.1,points=[[1.,.85,.9]])
        own=p.profile['fingers']['-1:index']
        from flowphi.motion import world_xy
        tip=world_xy(c['points'][0][1:],p.physical)
        root=tuple(tip[j]-own['offset'][j] for j in (0,1))
        for name in ('thumb','middle','ring','little'):
            p.profile['fingers'][f'-1:{name}'].update(base=(1.,1.),offset=(1.,1.))
        self.assertEqual(0.,visibility.contact(c))
        x,y=visibility.notes(1.05)[0][0][1]
        p.profile['fingers']['-1:middle'].update(base=(x-root[0],y-root[1]-.02),
                                               offset=(x-root[0],y-root[1]+.02))
        visibility.contacts.clear()
        self.assertGreater(visibility.contact(c),0.)

    def fixture(self, beat=1.3, kind=NoteType.TAP, hold=0.):
        note = SimpleNamespace(seconds=beat, type=kind, hold=hold)
        visual = SimpleNamespace(note=note, is_fake=False)
        line = SimpleNamespace(visual_notes=[visual], angle=SimpleNamespace())
        class Angle:
            def __matmul__(self, t):
                return 0.
        line.angle = Angle()
        line.note_state = lambda t, v: NoteState(500+200j, 500+400j, 1., 1., 1., True)
        task = SimpleNamespace(note=note, line=line)
        planner = SimpleNamespace(chart=SimpleNamespace(lines=[line]), original_tasks={0: task},
            screen=SimpleNamespace(width=1000, height=600), physical=(.28, .168), profile=default_profile())
        return planner, OcclusionScorer(planner)

    def test_future_note_obscured_by_finger_but_not_lower_contact(self):
        p, visibility = self.fixture()
        upper = dict(hand='left', finger='index', start=1., end=1.1, points=[[1., .5, 1/3]])
        lower = dict(upper, points=[[1., .5, .9]])
        self.assertGreater(visibility.contact(upper), 0.)
        self.assertEqual(0., visibility.contact(lower))

    def test_expired_and_distant_notes_are_not_penalized(self):
        for beat in (.9, 2.):
            _, visibility = self.fixture(beat)
            self.assertEqual([], visibility.notes(1.))

    def test_hidden_note_and_offscreen_note_do_not_contribute(self):
        p, visibility = self.fixture()
        p.chart.lines[0].note_state = lambda t, v: None
        self.assertEqual([], visibility.notes(1.))
        p, visibility = self.fixture()
        p.chart.lines[0].note_state = lambda t, v: NoteState(500-200j, 500-200j, 1., 1., 1., True)
        self.assertEqual([], visibility.notes(1.))

    def test_active_hold_body_is_still_protected(self):
        _, visibility = self.fixture(.8, NoteType.HOLD, 1.)
        self.assertTrue(visibility.notes(1.))
        # Body extends from y=200 to y=400; the middle remains visible.
        self.assertGreater(visibility.coverage(1., [((-.04, 0.), (.04, 0.), .015)]), 0.)

    def test_actual_palm_and_idle_fingers_count_without_active_down(self):
        p, visibility = self.fixture()
        point = visibility.notes(1.)[0][0][1]
        x, y = point
        frame = dict(hands={'left': dict(palm=[(x-.03,y-.02, .1), (x+.03,y-.02,.1),
            (x+.03,y+.02,.1), (x-.03,y+.02,.1)], fingers={})})
        self.assertGreater(visibility.frame(1., frame), 0.)
        frame['hands']['left']['palm'] = []
        frame['hands']['left']['fingers'] = {'index': dict(joints=[(x,y-.04,.1), (x,y+.04,.1)])}
        self.assertGreater(visibility.frame(1., frame), 0.)

    def test_duration_is_integrated_and_caches_are_bounded(self):
        _, visibility = self.fixture()
        a = dict(hand='left', finger='index', start=1., end=1.1, points=[[1., .5, 1/3]])
        b = dict(a, end=1.2)
        self.assertGreater(visibility.contact(b), visibility.contact(a))
        for i in range(MAX_CACHE+20):
            _remember(visibility.samples, i, [])
        self.assertEqual(MAX_CACHE, len(visibility.samples))

    def test_real_chart_and_shared_scorer_prefer_unobstructed_equal_judgement(self):
        p = Planner(make_chart([(1,1.,0.,0.), (1,1.2,0.,0.)]), Settings(), default_profile())
        # Both points lie on the same legal vertical judgement band. The upper
        # one covers an incoming note; the lower one leaves its approach clear.
        a = dict(hand='left', finger='index', start=1., end=1.08, kind='tap',
                 note_ids=[0], points=[[1., .5, .4]])
        b = dict(a, points=[[1., .5, .85]])
        for record in (a, b):
            score_contact(p, record, (), 0., 0., [])
        self.assertGreater(a['decision']['costs']['occlusion'], b['decision']['costs']['occlusion'])
        disabled = Planner(p.chart, Settings(occlusion_weight=0.), default_profile())
        score_contact(disabled, b, (), 0., 0., [])
        self.assertEqual(0., b['decision']['costs']['occlusion'])

    def test_weight_validation(self):
        for weight in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                Settings(occlusion_weight=weight)

    def test_mirrored_palm_vertices_use_silhouette_not_bow_tie(self):
        palm = [(-1.,0.,0.), (1.,0.,0.), (-1.,1.,0.), (1.,1.,0.)]
        self.assertTrue(_inside((.7,.7), _hull(palm)))
        self.assertTrue(_inside((-.7,.7), _hull(palm)))


if __name__ == '__main__':
    unittest.main()
