import numpy as np
from .metrics import boundary, boundary_score, score


def test_missing_boundary_is_zero_and_gt_controls_eligibility():
    truth = np.zeros((16,16), bool); truth[4:12,4:12] = True
    valid = np.ones_like(truth)
    for pred in (np.zeros_like(truth), np.ones_like(truth)):
        result = score(pred, truth, valid, [.5,.5])
        assert result['boundary_evaluable'] and result['bf1_um2'] == 0
    assert score(truth, truth, valid, [.5,.5])['bf1_um2'] == 1
    assert score(truth, np.zeros_like(truth), valid, [.5,.5])['bf1_um2'] is None


def test_euclidean_tolerance_excludes_square_diagonal_and_uses_anisotropy():
    a = np.zeros((10,10), bool); b = a.copy(); a[2,2] = True; b[5,5] = True
    assert boundary_score(a,b,[.5,.5])['bf1_um2'] == 0  # 2.12um > 2um
    b[:] = False; b[2,6] = True
    assert boundary_score(a,b,[.5,1.])['bf1_um2'] == 1
    assert boundary_score(a,b,[1.,.5])['bf1_um2'] == 0


def test_ignore_label_does_not_create_boundary():
    mask = np.zeros((8,8), bool); mask[:,:4] = True
    valid = np.ones_like(mask); valid[:,3:5] = False
    assert not boundary(mask,valid).any()


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(f) for f in (
        test_missing_boundary_is_zero_and_gt_controls_eligibility,
        test_euclidean_tolerance_excludes_square_diagonal_and_uses_anisotropy,
        test_ignore_label_does_not_create_boundary))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
