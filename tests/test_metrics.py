import numpy as np

from metrics import recall_at_k

# 12 ground-truth neighbours for one query. Distances 0..8 are distinct, but ids 9, 10 and 11
# are all tied at distance 9, so with k=10 the 10th neighbour is "any one of ids 9, 10, 11".
GT_IDS = np.array([np.arange(12)])
GT_DISTS = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]], dtype="float32")


def rec(found):
    return recall_at_k(np.array([found]), GT_IDS, GT_DISTS, k=10)


def test_perfect():
    assert rec(list(range(10))) == 1.0


def test_nothing_right():
    assert rec(list(range(100, 110))) == 0.0


def test_half_right():
    assert rec(list(range(5)) + list(range(100, 105))) == 0.5


def test_tie_at_10th_is_not_a_mistake():
    assert rec(list(range(9)) + [10]) == 1.0      # id 10 is as close as id 9
    assert rec(list(range(9)) + [11]) == 1.0


def test_missing_a_strictly_closer_neighbour_is_a_mistake():
    assert rec([0, 1, 2, 3, 4, 5, 6, 7, 10, 11]) == 0.9   # id 8 missing; only ONE tied id may fill a slot
    assert rec([0, 1, 2, 3, 4, 5, 6, 7, 9, 10]) == 0.9   # id 8 (distance 8) is missing


def test_repeated_ids_count_once():
    assert rec([0] * 10) == 0.1


def test_padding_minus_one_is_wrong():
    assert rec(list(range(9)) + [-1]) == 0.9


def test_average_over_queries():
    gt_ids = np.vstack([GT_IDS, GT_IDS])
    gt_d = np.vstack([GT_DISTS, GT_DISTS])
    found = np.array([list(range(10)), list(range(100, 110))])
    assert recall_at_k(found, gt_ids, gt_d, k=10) == 0.5
