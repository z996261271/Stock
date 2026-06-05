import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.execution.rebalance import normalize_current_weights, target_rebalance_weight  # noqa: E402


def test_rebalance_helpers_normalize_weights_and_apply_single_name_cap():
    symbols = np.asarray(["000001", "000002", "000003"], dtype=str)

    assert normalize_current_weights(symbols, None).tolist() == [1 / 3, 1 / 3, 1 / 3]
    assert normalize_current_weights(symbols, np.asarray([0.4, np.nan, -0.1])).tolist() == [0.4, 0.0, 0.0]
    assert target_rebalance_weight(symbols, None, 0.2) == 0.2
    assert target_rebalance_weight(symbols, 0.15, 0.2) == 0.15
    assert target_rebalance_weight(np.asarray([], dtype=str), None, 0.2) == 0.0


if __name__ == "__main__":
    test_rebalance_helpers_normalize_weights_and_apply_single_name_cap()
