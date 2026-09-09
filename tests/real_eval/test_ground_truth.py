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
    module_name = "real_eval.ground_truth"
    module_path = Path(__file__).parents[2] / "real_eval" / "ground_truth.py"
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
