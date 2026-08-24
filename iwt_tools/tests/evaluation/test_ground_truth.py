"""Регрессионные тесты строгого сопоставления имён Roboflow."""

from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType


def _load_ground_truth_module() -> ModuleType:
    """Загружает независимую утилиту без тяжёлых импортов inference-пакета."""

    if find_spec("PIL") is None:
        pil_stub = ModuleType("PIL")
        pil_stub.Image = ModuleType("PIL.Image")
        sys.modules["PIL"] = pil_stub
    module_name = "iwt_tools.evaluation.ground_truth"
    module_path = Path(__file__).parents[2] / "src" / "iwt_tools" / "evaluation" / "ground_truth.py"
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ground_truth = _load_ground_truth_module()


def test_roboflow_canonical_keys_match_real_filename() -> None:
    """Проверяет реальное переименование точек, расширения и hash в Roboflow."""

    source_name = "26_781_az_171.350_el_11.900.png"
    coco_name = (
        "26_781_az_171-350_el_11-900_png"
        ".rf.17r7IB1B25o1FlWTwFTZ.png"
    )
    expected_key = "26_781_az_171-350_el_11-900_png"

    source_key = ground_truth.canonical_source_key(source_name)
    coco_key = ground_truth.canonical_coco_key(coco_name)

    assert source_key == expected_key
    assert coco_key == expected_key
    assert source_key == coco_key


def test_roboflow_canonical_match_selects_exactly_one_source(tmp_path: Path) -> None:
    """Проверяет выбор единственного файла через третий canonical-способ."""

    source_path = tmp_path / "26_781_az_171.350_el_11.900.png"
    source_path.touch()
    coco_name = (
        "26_781_az_171-350_el_11-900_png"
        ".rf.17r7IB1B25o1FlWTwFTZ.png"
    )

    candidates = ground_truth._match_candidates(
        coco_name,
        tmp_path,
        (source_path,),
    )

    assert candidates == (source_path,)


def test_roboflow_canonical_match_does_not_guess_missing_source(
    tmp_path: Path,
) -> None:
    """Подтверждает отсутствие fuzzy matching при похожем, но другом имени."""

    similar_path = tmp_path / "26_781_az_171.351_el_11.900.png"
    similar_path.touch()
    coco_name = (
        "26_781_az_171-350_el_11-900_png"
        ".rf.17r7IB1B25o1FlWTwFTZ.png"
    )

    candidates = ground_truth._match_candidates(
        coco_name,
        tmp_path,
        (similar_path,),
    )

    assert candidates == ()


def test_category_mapping_allows_unused_service_category() -> None:
    """Проверяет явный mapping bpla/uncertain и неиспользуемый root-класс."""

    categories = [
        {"id": 0, "name": "bpla-only-small-target"},
        {"id": 1, "name": "bpla"},
        {"id": 2, "name": "uncertain_bpla"},
    ]

    assert ground_truth._build_category_index(categories) == {
        1: ("bpla", "scored"),
        2: ("uncertain_bpla", "uncertain"),
    }


def test_annotation_cannot_use_service_category() -> None:
    """Проверяет строгую ошибку при annotation неизвестной роли."""

    images = {1: {"id": 1, "width": 10, "height": 10}}
    annotations = [{
        "id": 7, "image_id": 1, "category_id": 0,
        "bbox": [1, 1, 2, 2], "segmentation": [],
    }]

    try:
        ground_truth._validate_annotations(annotations, images, {})
    except ground_truth.AuditError as error:
        assert "unsupported category_id 0" in str(error)
    else:
        raise AssertionError("Unsupported annotation category was accepted")


def test_rows_keep_empty_images_and_exclude_uncertain_from_scored_counts() -> None:
    """Проверяет empty frame и раздельные scored/uncertain счётчики."""

    matches = {
        1: ground_truth.ImageMatch(
            1, "scored.png", ground_truth.MatchStatus.MATCH, Path("scored.png"),
            "scored.png", 10, 10, 10, 10,
        ),
        2: ground_truth.ImageMatch(
            2, "uncertain.png", ground_truth.MatchStatus.MATCH,
            Path("uncertain.png"), "uncertain.png", 10, 10, 10, 10,
        ),
        3: ground_truth.ImageMatch(
            3, "empty.png", ground_truth.MatchStatus.MATCH, Path("empty.png"),
            "empty.png", 10, 10, 10, 10,
        ),
    }
    audit = ground_truth.AuditResult(
        coco_path=Path("_annotations.coco.json"), positive_frames=3,
        images=(
            {"id": 1, "file_name": "scored.png", "width": 10, "height": 10},
            {"id": 2, "file_name": "uncertain.png", "width": 10, "height": 10},
            {"id": 3, "file_name": "empty.png", "width": 10, "height": 10},
        ),
        annotations=(
            {"id": 1, "image_id": 1}, {"id": 2, "image_id": 2},
        ),
        matches=matches,
        annotations_by_image={
            1: ({"id": 1, "bbox": [1, 1, 2, 2], "category_name": "bpla",
                 "gt_role": "scored"},),
            2: ({"id": 2, "bbox": [1, 1, 2, 2],
                 "category_name": "uncertain_bpla", "gt_role": "uncertain"},),
        },
        images_without_annotations=("empty.png",),
    )

    object_rows, image_rows = ground_truth._build_rows(audit)
    by_name = {row.image: row for row in image_rows}

    assert [row.gt_role for row in object_rows] == ["scored", "uncertain"]
    assert object_rows[1].size_class == ""
    assert by_name["scored.png"].gt_target_count == 1
    assert by_name["uncertain.png"].uncertain_target_count == 1
    assert by_name["empty.png"].gt_target_count == 0
    assert by_name["empty.png"].uncertain_target_count == 0
