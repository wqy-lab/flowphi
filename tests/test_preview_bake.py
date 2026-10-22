import unittest

from flowphi.rendering.blender import sample_times


class PreviewSamplingTests(unittest.TestCase):
    def job(self):
        return dict(start=0., duration=1., frames=15, fps=15, warmup=1.,
            contacts=[dict(start=.123, end=.789, beat=.15, prepare=.05,
                release_until=.9, points=[[.123, .2, .3], [.456, .4, .5], [.789, .6, .7]])])

    def test_preview_keeps_output_frames_and_touch_boundaries(self):
        times = sample_times(dict(self.job(), preview_sampling=True))
        self.assertEqual(times, sorted(set(times)))
        for when in [.123, .789, .15, .05, .9, *[i / 15 for i in range(15)]]:
            self.assertLess(min(abs(t - when) for t in times), 1e-8)

    def test_detailed_sampling_still_keeps_trajectory_knots(self):
        job = self.job()
        self.assertIn(.456, sample_times(job))
        self.assertNotIn(.456, sample_times(dict(job, preview_sampling=True)))
        self.assertEqual(sample_times(job), sample_times(dict(job, preview_sampling=False)))


if __name__ == '__main__':
    unittest.main()
