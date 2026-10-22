"""High-top radiation always retains one clear buffer above the model lid."""
import unittest

from gpuwm.core.rrtmg_legacy_prep import compute_lw_nlayers


class HighTopLongwaveLayers(unittest.TestCase):
    def test_high_top_keeps_all_model_layers_and_one_clear_buffer(self):
        for levels in (55, 75, 80, 90):
            for p_top in (1., 10., 100., 199.):
                with self.subTest(levels=levels, p_top=p_top):
                    self.assertEqual(compute_lw_nlayers(levels + 1, p_top), levels + 1)

    def test_ordinary_top_retains_source_buffer_count(self):
        # WRF rounds p_top / 4 hPa to the nearest integer, half up.
        for p_top, extras in ((200., 1), (400., 1), (600., 2), (3000., 8)):
            with self.subTest(p_top=p_top):
                self.assertEqual(compute_lw_nlayers(56, p_top), 55 + extras)


if __name__ == '__main__':
    unittest.main()
