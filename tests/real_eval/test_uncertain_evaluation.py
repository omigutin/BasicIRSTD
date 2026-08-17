"""Тесты общей scored/uncertain семантики object-level evaluation."""

from importlib.util import find_spec

import pytest

HAS_MATCHING_DEPS = find_spec("numpy") is not None and find_spec("scipy") is not None

from real_eval.evaluation_core import (
    GroundTruthFrame,
    GroundTruthRecord,
    GroundTruthRole,
    PredictedComponent,
    has_all_scored_targets_detected,
    match_frame_predictions,
)


def _gt(annotation_id: int, role: GroundTruthRole, x1: float = 10) -> GroundTruthRecord:
    """Создаёт scored или uncertain bbox для компактных сценариев."""

    return GroundTruthRecord(
        image="frame.png", annotation_id=annotation_id,
        size_class="Tiny" if role is GroundTruthRole.SCORED else None,
        x1=x1, y1=10, x2=x1 + 4, y2=14,
        category_name="bpla" if role is GroundTruthRole.SCORED else "uncertain_bpla",
        role=role,
    )


def _prediction(index: int, x: float, y: float = 12) -> PredictedComponent:
    """Создаёт prediction с заданным centroid."""

    return PredictedComponent(index, index, int(x), int(y), int(x) + 1, int(y) + 1,
                              x, y, 1, 1.0)


@pytest.mark.skipif(not HAS_MATCHING_DEPS, reason="NumPy/SciPy are unavailable")
def test_a_scored_prediction_is_true_positive() -> None:
    """A: обычный BPLA имеет стандартную TP/FN/FP семантику."""

    result = match_frame_predictions(
        GroundTruthFrame("frame.png", (_gt(1, GroundTruthRole.SCORED),), ()),
        (_prediction(1, 12),),
    )
    assert (result.tp, result.fn, result.fp, len(result.ignored_uncertain)) == (1, 0, 0, 0)


def test_b_uncertain_prediction_is_ignored() -> None:
    """B: prediction возле uncertain не получает TP или FP."""

    result = match_frame_predictions(
        GroundTruthFrame("frame.png", (), (_gt(2, GroundTruthRole.UNCERTAIN),)),
        (_prediction(1, 12),),
    )
    assert (result.tp, result.fn, result.fp, len(result.ignored_uncertain)) == (0, 0, 0, 1)


def test_c_unmatched_uncertain_does_not_create_fn() -> None:
    """C: uncertain без prediction не создаёт FN."""

    result = match_frame_predictions(
        GroundTruthFrame("frame.png", (), (_gt(2, GroundTruthRole.UNCERTAIN),)), ()
    )
    assert (result.tp, result.fn, result.fp) == (0, 0, 0)


def test_d_prediction_far_from_uncertain_is_false_positive() -> None:
    """D: дальний prediction остаётся настоящим FP."""

    result = match_frame_predictions(
        GroundTruthFrame("frame.png", (), (_gt(2, GroundTruthRole.UNCERTAIN),)),
        (_prediction(1, 30),),
    )
    assert (result.tp, result.fn, result.fp, len(result.ignored_uncertain)) == (0, 0, 1, 0)


@pytest.mark.skipif(not HAS_MATCHING_DEPS, reason="NumPy/SciPy are unavailable")
def test_e_scored_match_precedes_uncertain_ignore() -> None:
    """E: разные predictions дают scored TP и uncertain IGNORE."""

    frame = GroundTruthFrame(
        "frame.png", (_gt(1, GroundTruthRole.SCORED),),
        (_gt(2, GroundTruthRole.UNCERTAIN, x1=30),),
    )
    result = match_frame_predictions(frame, (_prediction(1, 12), _prediction(2, 32)))
    assert (result.tp, result.fn, result.fp, len(result.ignored_uncertain)) == (1, 0, 0, 1)


def test_f_ignore_only_frame_has_no_scored_targets() -> None:
    """F: ignore-only определяется наличием uncertain при отсутствии scored."""

    frame = GroundTruthFrame("frame.png", (), (_gt(2, GroundTruthRole.UNCERTAIN),))
    assert frame.is_ignore_only
    assert not frame.is_scored_positive
    assert not has_all_scored_targets_detected(0, 0)
    result = match_frame_predictions(frame, ())
    assert result.tp == 0 and result.fn == 0


@pytest.mark.skipif(not HAS_MATCHING_DEPS, reason="NumPy/SciPy are unavailable")
def test_g_scored_ground_truth_has_priority_over_uncertain() -> None:
    """G: один prediction сначала становится TP, а не uncertain IGNORE."""

    frame = GroundTruthFrame(
        "frame.png", (_gt(1, GroundTruthRole.SCORED),),
        (_gt(2, GroundTruthRole.UNCERTAIN),),
    )
    result = match_frame_predictions(frame, (_prediction(1, 12),))
    assert (result.tp, result.fp, len(result.ignored_uncertain)) == (1, 0, 0)
