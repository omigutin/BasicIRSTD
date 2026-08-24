# IWT Tools

`iwt_tools` — внутренний Python-пакет и набор инструментов IWT для оценки
моделей BasicIRSTD, подготовки CIX и сравнения результатов. Исходные модели
BasicIRSTD, наборы данных, рабочие результаты и checkpoints остаются в корне
основного репозитория.

## Установка

Из корня BasicIRSTD выполнить:

```bash
python -m pip install -e ./iwt_tools
```

Editable-установка добавляет `src/iwt_tools` в Python-окружение. Запуск из корня
BasicIRSTD также даёт `ModelRunner` доступ к upstream-модулям `net.py`, `utils.py` и
`model/` без изменения `sys.path`.

## Фактическая структура

| Каталог | Назначение |
| --- | --- |
| `src/iwt_tools/evaluation` | Benchmark, ground truth, matching и оценка моделей |
| `src/iwt_tools/cix` | Выбор CIX-данных и сравнение PyTorch ↔ CIX |
| `scripts/conversion` | Экспорт ONNX и подготовка CIX calibration |
| `scripts/validation` | Подготовка и сравнение validation-результатов PyTorch и CIX |
| `scripts/orangepi` | Запуск CIX на NPU Orange Pi и замер производительности |
| `scripts/data` | Вспомогательная подготовка данных |
| `scripts/training` | Аудит IRSTD-1K, подготовка YOLO Detect и обучение YOLO |
| `models/alcnet_irstd1k` | Проверенные Batch 1 ONNX/CIX и отдельный каталог `batch4` для Batch 4 артефактов |
| `data/ground_truth` | Раздельные ground truth `iwt_device_all_bpla` и `iwt_device_all_bpla2` |
| `tests` | Pytest-тесты IWT-кода |

## Запускаемые инструменты

### Python-модули

- `python -m iwt_tools.evaluation.main` — оценка BasicIRSTD на изображениях или видео.
- `python -m iwt_tools.evaluation.ground_truth` — аудит COCO-разметки и подготовка ground truth.
- `python -m iwt_tools.evaluation.evaluate_model` — benchmark PyTorch-модели.
- `python -m iwt_tools.evaluation.evaluate_yolo` — benchmark YOLO-модели.
- `python -m iwt_tools.evaluation.evaluate_classical` — benchmark классического метода.

### Скрипты

- `iwt_tools/scripts/conversion/export_alcnet_to_onnx.py` — экспорт ALCNet / IRSTD-1K в статический ONNX.
- `iwt_tools/scripts/conversion/prepare_cix_calibration.py` — подготовка `calibration.npy` для CixBuilder.
- `iwt_tools/scripts/validation/export_cix_validation.py` — подготовка validation inputs и PyTorch outputs.
- `iwt_tools/scripts/validation/compare_cix_outputs.py` — сравнение результатов PyTorch и CIX.
- `iwt_tools/scripts/orangepi/run_cix_validation.py` — запуск `.cix` на Orange Pi NPU и замер производительности.
- `iwt_tools/scripts/orangepi/deploy_cix_validation.sh` — копирование runner, модели и input на Orange Pi.
- `iwt_tools/scripts/data/sample_video_frames.py` — выборка PNG-кадров из видео с заданным интервалом.
- `iwt_tools/scripts/training/prepare_irstd1k_yolo.py` — аудит масок и подготовка YOLO Detect.
- `iwt_tools/scripts/training/train_yolo.py` — общее обучение Detect-моделей Ultralytics YOLO.

Все пути к наборам данных, результатам, ground truth и checkpoints передаются через
аргументы соответствующего CLI. Скрипты следует запускать из корня BasicIRSTD
после editable-установки пакета.

> **Перед повторением конвертации сначала подтвердите актуальные пути и имена
> файлов.** Пути к проекту, датасетам, WSL, Orange Pi и моделям не являются постоянным
> контрактом.

## IRSTD-1K для YOLO Detect

Установите в локальное окружение совместимые версии OpenCV, NumPy, PyTorch и
Ultralytics. Исходный `datasets/IRSTD-1K` скрипт только читает. Подготовка всегда
использует исходные `train_IRSTD-1K.txt` и `test_IRSTD-1K.txt`, не выполняя
случайное разбиение:

```powershell
python iwt_tools/scripts/training/prepare_irstd1k_yolo.py `
    --dataset datasets/IRSTD-1K `
    --output datasets/IRSTD-1K-YOLO
```

До записи результата скрипт проверяет все пары image/mask и печатает значения
масок, пустые маски, connected components (связные области), Tiny-объекты и
статистику рамок. Если маска содержит несколько ненулевых значений, подготовка
останавливается. Флаг `--allow-multivalue-masks` допустим только после ручной
проверки, что каждое ненулевое значение действительно обозначает цель.

После подготовки просмотрите PNG-файлы в
`datasets/IRSTD-1K-YOLO/dataset_check/`.

Обучение YOLO26n:

```powershell
python iwt_tools/scripts/training/train_yolo.py `
    --model yolo26n.pt `
    --dataset datasets/IRSTD-1K-YOLO/dataset.yaml `
    --output iwt_tools/models/yolo26_irstd1k
```

Обучение YOLO8n при необходимости:

```powershell
python iwt_tools/scripts/training/train_yolo.py `
    --model yolov8n.pt `
    --dataset datasets/IRSTD-1K-YOLO/dataset.yaml `
    --output iwt_tools/models/yolo8_irstd1k
```

Без CUDA обучение не начинается. Флаг `--allow-cpu` разрешает CPU-обучение
только по явному решению пользователя. Лучший checkpoint сохраняется как
`<output>/best.pt`, а полный журнал — в `<output>/training/`.

## ALCNet Batch 1 и Batch 4

ALCNet / IRSTD-1K поддерживает статические контракты `[1, 1, 640, 512]` и
`[4, 1, 640, 512]`. Проверенные Batch 1 артефакты остаются непосредственно в
`models/alcnet_irstd1k`, а Batch 4 ONNX, CIX, отчёт и `build.cfg` располагаются в
`models/alcnet_irstd1k/batch4`. Validation inputs и CIX outputs в обоих случаях
хранятся по кадрам как `[N, 1, 640, 512]`.

Экспорт статического Batch 4 ONNX:

```bash
python iwt_tools/scripts/conversion/export_alcnet_to_onnx.py \
  --batch-size 4 --image <REAL_IMAGE> --height 640 --width 512 \
  --output-dir iwt_tools/models/alcnet_irstd1k/batch4
```

В рабочем окружении CixBuilder оставьте ONNX рядом с `build.cfg`, а неизменённый
per-frame `calibration.npy` положите в указанный config каталог `datasets`, затем
из каталога `batch4` выполните:

```bash
cd iwt_tools/models/alcnet_irstd1k/batch4
cixbuild build.cfg
```

Запуск Batch 4 CIX на Orange Pi:

```bash
python iwt_tools/scripts/orangepi/run_cix_validation.py \
  --batch-size 4 --model iwt_tools/models/alcnet_irstd1k/batch4/alcnet_irstd1k.cix \
  --input <VALIDATION_DIR>/input.npy --output <VALIDATION_DIR>/cix_outputs_batch4.npy
```

## CIX validation на Orange Pi

Vendor binding должен быть установлен в Python-окружении Orange Pi и доступен
как `NOE_Engine`. Из корня BasicIRSTD скопируйте автономный runner, модель и input:

```bash
iwt_tools/scripts/orangepi/deploy_cix_validation.sh \
  orangepi '~/cix-validation' \
  iwt_tools/models/istdunet_irstd1k/istdunet_irstd1k.cix \
  iwt_tools/data/cix_validation/istdunet_irstd1k/input.npy
```

На Orange Pi запустите validation из Python-окружения с установленным runtime:

```bash
~/projects/CIXInference/.venv/bin/python ~/cix-validation/run_cix_validation.py \
  --model ~/cix-validation/istdunet_irstd1k.cix \
  --input ~/cix-validation/input.npy \
  --output ~/cix-validation/cix_outputs.npy \
  --batch-size 1 --warmup 20
```
