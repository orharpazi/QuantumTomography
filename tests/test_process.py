import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "process",
    Path(__file__).resolve().parents[1] / "src" / "process.py",
)
process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process)


def test_parse_angle_pair_scales_large_angle_values_to_degrees():
    assert process.parse_angle_pair("-45,225") == (-45.0, 22.5)
    assert process.parse_angle_pair("0,675") == (0.0, 67.5)
    assert process.parse_angle_pair("90,1125") == (90.0, 112.5)


def test_set_threshold_uses_each_series_maximum_for_column_major_input():
    data = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=float,
    )

    assert np.allclose(process.set_threshold(data), np.array([1.2, 1.5], dtype=float))


def test_calculate_n_does_not_emit_debug_output(capsys):
    data = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.8, 0.2, 0.2, 0.8],
            [0.2, 0.2, 0.8, 0.8, 0.2, 0.2],
        ],
        dtype=float,
    )
    threshold = np.array([0.5, 0.5], dtype=float)

    assert process.calculate_n(data, threshold, total_pairs=3) == (1, 1, 0, 1)
    assert capsys.readouterr().out == ""


def test_lookup_uses_normalized_angle_keys_for_chsh_results():
    results = {(0.0, 22.5): 0.75}

    assert process.get_chsh_value(results, (0.0, 225.0)) == 0.75
