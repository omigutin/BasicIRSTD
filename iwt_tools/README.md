# IWT Tools

Внутренние инструменты IWT для оценки моделей BasicIRSTD, подготовки данных CIX и
проверки результатов. Исходные модели BasicIRSTD, наборы данных, результаты и
контрольные точки остаются в корне основного репозитория.

Установить модуль для разработки:

```bash
python -m pip install -e ./iwt_tools
```

## Структура

| Каталог | Назначение |
| --- | --- |
| `src/iwt_tools/evaluation` | Общая логика benchmark, ground truth и оценки моделей |
| `src/iwt_tools/cix` | Выбор данных и сравнение результатов CIX |
| `scripts/conversion` | Подготовка данных для преобразования PyTorch/ONNX/CIX |
| `scripts/validation` | Подготовка и сравнение результатов PyTorch и CIX |
| `scripts/orangepi` | Запуск CIX на Orange Pi |
| `models` | Место для финальных ONNX- и CIX-моделей; артефакты в Git сейчас отсутствуют |
| `tests` | Регрессионные тесты пользовательского IWT-кода |

## Запускаемые файлы

- `python -m iwt_tools.evaluation.main` — оценка BasicIRSTD на изображениях или видео.
- `python -m iwt_tools.evaluation.ground_truth` — аудит COCO-разметки и подготовка ground truth.
- `python -m iwt_tools.evaluation.evaluate_model` — benchmark PyTorch-модели.
- `python -m iwt_tools.evaluation.evaluate_yolo` — benchmark YOLO-модели.
- `python -m iwt_tools.evaluation.evaluate_classical` — benchmark классического метода.
- `scripts/conversion/prepare_cix_calibration.py` — подготовка `calibration.npy` для CixBuilder.
- `scripts/validation/export_cix_validation.py` — подготовка validation inputs и PyTorch outputs.
- `scripts/validation/compare_cix_outputs.py` — сравнение результатов PyTorch и CIX.
- `scripts/orangepi/run_cix_validation.py` — массовый запуск CIX на Orange Pi и замер производительности.

Все пути к наборам данных, результатам, ground truth и checkpoints передаются через
аргументы соответствующего CLI. Команды следует запускать из корня BasicIRSTD после
editable-установки пакета.
