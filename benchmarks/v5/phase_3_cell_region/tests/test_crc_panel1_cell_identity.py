import numpy as np

from benchmarks.v5.tools.run_crc_panel1_cell_identity import label, score, spatial_split


def test_spatial_block_gap_and_identity_mapping():
    assert spatial_split(0, 0) is None
    assert spatial_split(512, 512) == spatial_split(3072, 3072)
    assert spatial_split(3584, 512) is None
    assert label('Epithelial_S100') == 0
    assert label('Blood_endothelial_S100') == 1
    assert label('FAP_SMA_CAF') == 3
    assert label('Undefined') == -1
    assert label('D2-40_positive') == -1


def test_per_class_report_exposes_majority_collapse():
    truth = np.array([0, 0, 1, 2, 3, 4])
    result = score(truth, np.zeros(len(truth), dtype=int))
    assert result['per_class'][1]['support'] == 1
    assert result['per_class'][1]['recall'] == 0
    assert result['macro_f1'] < .2
