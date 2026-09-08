# Фактические контракты моделей BasicIRSTD для CIXInference

Дата аудита: 2026-09-08. Область аудита: три логические модели ALCNet IRSTD-1K,
YOLO26 IRSTD-1K и YOLO26 IRSTD-1K IWT. Это отчёт о существующем состоянии:
архитектура, веса и runtime не изменялись.

Термины: **сырой выход** ниже означает результат backend до порога, NMS и
формирования объектов CIXInference. Там, где Ultralytics high-level `predict()` уже
делает postprocessing, отдельно указан выход самого `DetectionModel`/экспортного
графа и выход API.

## A. Таблица моделей

| Model | PyTorch artifact | Тип checkpoint | Architecture | CPU | CUDA |
|---|---|---|---|---|---|
| ALCNet IRSTD-1K | Ожидается `log/IRSTD-1K/ALCNet_400.pth.tar`, но файла в checkout **нет**; SHA-256 использованного при экспорте файла: `93a661de…b7b0` | checkpoint `dict` с обязательным `state_dict`; не полный module | `net.Net(model_name="ALCNet", mode="test")` → `model.ACM.model_ALCnet.ASKCResNetFPN` | Поддержан через `map_location` и `ModelRunner`; в этой среде не запущен из-за отсутствующих checkpoint и Python dependencies | Поддержан тем же кодом при `torch.cuda.is_available()`; здесь CUDA/PyTorch отсутствуют |
| YOLO26 IRSTD-1K | `iwt_tools/models/yolo26_irstd1k/yolo26_irstd1k.pt` | Ultralytics training checkpoint `dict`, внутри ключа `model` сериализован полный `ultralytics.nn.tasks.DetectionModel` (half weights), плюс epoch/EMA/train args/metrics; это не простой `state_dict` | Ultralytics **YOLO26n**, `DetectionModel`, YAML `yolo26n.yaml`, Detect head | Поддержан `ultralytics.YOLO(...).predict(device="cpu")`; не запущен: пакеты отсутствуют | Поддержан тем же API с `device="cuda"`/`0`; здесь не проверен |
| YOLO26 IRSTD-1K IWT | `iwt_tools/models/yolo26_irstd1k_iwt/yolo26_irstd1k_iwt.pt` | Такой же Ultralytics checkpoint с полным `DetectionModel`; второй этап стартовал от первого `.pt` | Та же YOLO26n/`yolo26n.yaml`; отличаются обученные веса и dataset metadata | Да, общий adapter | Да, общий adapter; здесь не проверен |

Проверка ZIP/pickle без исполнения показала в обоих `.pt` ссылки на
`ultralytics.nn.tasks.DetectionModel`, `ultralytics.nn.modules.head.Detect`, C3k2,
C3k, C2PSA, SPPF, Conv/DWConv/Concat и ключи `model`, `train_args`, `epoch`,
`optimizer`. Следовательно, `.pt` не самостоятельны: unpickle требует совместимую
версию Ultralytics с YOLO26-классами. Исходники YOLO-архитектуры в BasicIRSTD не
скопированы.

## B. ALCNet contract

```text
load:
  Net("ALCNet", "test")
  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint["state_dict"], strict=True)
  model.eval(); model.to(device)
input:
  float32 NCHW [B, 1, Hpad, Wpad]; проверенные CIX размеры
  B=1/4, Hpad=640, Wpad=512
preprocessing:
  PIL image.convert("I") -> float32 grayscale -> (x-87.4661865234375)/
  39.71953201293945 -> zero-pad справа/снизу до кратности 32 -> add C,B axes
raw output:
  один float32 Tensor [B, 1, Hpad, Wpad], probability map в [0,1]
postprocessing:
  crop к исходным H,W -> map > 0.5 -> 8-connected components -> bbox/stats;
  confidence компоненты = max probability среди её пикселей
dependencies:
  inference core: torch, torchvision; существующий loader/preprocessor также
  импортирует numpy, Pillow, OpenCV; текущий широкий import graph требует ещё
  matplotlib и scikit-image
files needed:
  model/ACM/model_ALCnet.py, model/ACM/fusion.py, model/__init__.py (либо узкий
  перенос только ALCNet imports), net.py, utils.py и отсутствующий
  ALCNet_400.pth.tar
```

### Источник и forward

`model/__init__.py` экспортирует `ASKCResNetFPN` как `ALCNet`; `net.Net` только
выбирает этот класс и делегирует `forward()`. Финал `ASKCResNetFPN.forward()` —
`return out.sigmoid()`. Поэтому результат уже probability map, дополнительный
sigmoid **запрещён**. NMS, bbox decode и objectness внутри ALCNet отсутствуют.
Внутренние `torchvision.transforms.Resize` выравнивают feature maps; это часть
сети, а не image preprocessing.

Проверенный `ModelRunner` принимает 2D grayscale, RGB или RGBA NumPy image. RGB/A
сначала проходит PIL `convert("I")`; это не OpenCV BGR grayscale formula. После
нормализации `PadImg` добавляет нули только снизу/справа. Для фактического export
кадр уже обязан иметь точный размер; resize исходного изображения снаружи
ALCNet не выполнялся. В отчётах экспорта оси записаны `[B,1,640,512]`, то есть
в данном наборе `H=640`, `W=512`, несмотря на имена CLI constants `height=512`,
`width=640`; CIX config и validation code подтверждают именно `[B,1,640,512]`.

`ModelRunner.predict()` считает простой `connectedComponents` для количества,
а bbox evaluation — `cv2.connectedComponentsWithStats(..., connectivity=8)`.
Порог — строгий `> 0.5`, площадь не фильтруется, bbox полуоткрытый
`[x, y, x+width, y+height]`, score компоненты — максимум карты по её label.
Текущая ALCNet CIX validation использует тот же preprocessed tensor, тот же
probability-map contract и тот же downstream `PredictionResult`/component helper.
NPU runner, однако, специализирован только на одном flattened ALCNet output и
не является YOLO runner.

### Минимальный существующий PyTorch path

```python
runner = ModelRunner(ModelConfig(
    model_name="ALCNet",
    checkpoint_path=checkpoint,
    train_dataset_name="IRSTD-1K",
    dataset_dir=Path("datasets"),
    threshold=0.5,
    device="cpu",  # либо "cuda"
))
tensor, original_hw = runner.preprocess(image)
raw = runner.run_preprocessed(tensor)  # already sigmoid, [1,1,Hpad,Wpad]
```

CPU и CUDA алгоритмически идентичны. Отличия — `map_location`, `.to(device)` и
CUDA synchronization при измерении времени. Старый корневой `inference.py`
фактически не является CPU-safe: создаёт model/input через `.cuda()` даже если
fallback load выбрал CPU. Для переноса следует брать подтверждённый
`iwt_tools.evaluation.model_runner.ModelRunner`, а не этот legacy CLI.

### ALCNet conversion

```text
log/IRSTD-1K/ALCNet_400.pth.tar (state_dict; отсутствует в checkout)
  -> Net/ASKCResNetFPN eval CPU
  -> deepcopy + только на export временная замена torchvision Resize на
     F.interpolate(bilinear, align_corners=False, antialias=False)
  -> torch.onnx.export(opset=17, constant folding, dynamo=False,
     names input/probability_map, static [1|4,1,640,512])
  -> CixBuilder INT8 calibration -> .cix
```

Перед ONNX скрипт сравнивает штатный Torch и export-friendly Torch, затем ONNX
Runtime. Оба сохранённых отчёта имеют status `ok`; граф имеет один output,
включает Sigmoid, не включает threshold/connected components. CIX получает тот
же логический probability-map output, но может численно отличаться из-за INT8.
Batch-1 CIX config лежит отдельно и ссылается на прежний путь/имя output
`alcnet_irstd1k_v2.cix`; batch-4 config находится рядом с artifact. Это риск
воспроизводимости путей, не различие контракта.

## C. YOLO26 IRSTD-1K contract

```text
load:
  from ultralytics import YOLO
  model = YOLO(".../yolo26_irstd1k.pt")
  model.predict(source=bgr, conf=0.25, iou=0.7, imgsz=640,
                device="cpu"|"cuda", classes=None, verbose=False)
input to project API:
  NumPy uint8-like HxW, HxWx1, or RGB HxWx3
preprocessing inside Ultralytics predictor:
  project converts gray -> BGR or RGB -> BGR; Ultralytics letterbox/resize,
  BGR->RGB, float conversion, /255, BCHW are internal to predict()
raw exported/model output:
  float32 [B, 5, 6720] for static 512x640: [cx,cy,w,h,score] along axis 1;
  coordinates decoded to input pixels, score already sigmoid; one class
high-level output:
  Results.boxes.xyxy/conf/cls after threshold and NMS, scaled to source image
postprocessing:
  confidence 0.25 and IoU-NMS 0.7 are predict arguments; project clips xyxy to
  source bounds and drops non-positive boxes
classes:
  one Detect class (class id 0); code supports optional class_id filter
```

The 6720 positions follow the exported Detect strides 8/16/32 at 512x640:
`64*80 + 32*40 + 16*20`. The ordinary ONNX has input `images`, one output
`output0`, export metadata `dynamic=False`, `opset=17`, `nms=False`; batch 1 was
simplified, batch 4 was not. The graph contains bbox decode and score Sigmoid,
but no threshold/NMS. It therefore does **not** output final detections.

There are two valid levels of PyTorch use:

1. The project's confirmed production-like path is `YOLO(...).predict(...)`. It
   includes preprocessing, decode, confidence filtering, NMS and coordinate
   scaling; its “raw result” for project code is already postprocessed Boxes.
2. Calling the underlying `DetectionModel` on an already normalized BCHW tensor
   is required to align with ONNX/CIX raw outputs. That lower-level adapter is
   not implemented in this repository and its exact return container varies
   with Ultralytics train/eval/export mode. CIXInference must pin and test the
   selected Ultralytics version rather than assume the high-level Results object
   equals `[B,5,6720]`.

### Ordinary ONNX versus `_xy_wh_scores.onnx`

The script `split_yolo_output_xy_wh.py` traces the final graph:

- ordinary output is final `Concat(axis=1)` of decoded bbox `[B,4,6720]` and
  sigmoid scores `[B,1,6720]`, hence `[B,5,6720]`;
- transformed graph removes the bbox concat and common final bbox `Mul`;
- it exposes three outputs: `boxes_xy [B,2,6720]`, `boxes_wh [B,2,6720]`, and
  the existing sigmoid score tensor `[B,1,6720]`;
- XY and WH are each multiplied by stride independently. Concatenating the
  three outputs along channel axis is verified bit-identical to ordinary ONNX.

All four CIX build configs explicitly consume the `_xy_wh_scores.onnx` variant,
with blank `detection_postprocess`; thus CIX has three raw outputs and no NMS.
The checked-in `.cix` filenames do not match the config's declared `_xy_wh.cix`
name, so provenance/rename must be confirmed before claiming byte-for-byte build
reproducibility.

### YOLO conversion

No YOLO export command/script is committed. ONNX embedded metadata proves the
resulting settings:

```text
yolo26_irstd1k.pt
  -> Ultralytics ONNX export, imgsz=[512,640], opset=17, dynamic=False,
     nms=False, batch=1 simplify=True / batch=4 simplify=False
  -> split_yolo_output_xy_wh.py
  -> CixBuilder configs, INT8 per-tensor activations/per-channel weights
  -> yolo26_irstd1k_b1.cix / _b4.cix (apparent renamed outputs)
```

The precise invocation, Ultralytics version, exporter version, calibration file
and rename step are not present and remain unknown.

## D. YOLO26 IRSTD-1K IWT contract

Contract is identical to section C. The checkpoint's embedded YAML is still
`yolo26n.yaml`, and pickle global/module inventory matches the first model. The
training script/README show the IWT model is fine-tuned from
`yolo26_irstd1k.pt`, not built as a new architecture. Only weights, dataset path
and training metadata differ. Both models can and should use one YOLO adapter,
with artifact path/name supplied by configuration.

Its pipeline is correspondingly:

```text
yolo26_irstd1k.pt -> further Ultralytics train on IRSTD-1K-IWT dataset
  -> yolo26_irstd1k_iwt.pt
  -> same static ONNX export settings
  -> same three-way XY/WH/score graph split
  -> corresponding CixBuilder config -> checked-in b1/b4 CIX
```

## E. Conversion pipeline summary

| Model | PT → ONNX | Graph transform | ONNX → CIX | Raw-output compatibility |
|---|---|---|---|---|
| ALCNet | committed exporter; opset 17; static B1/B4 `[B,1,640,512]` | export-only Resize implementation, validated against original | CixBuilder, calibrated INT8 | Same one probability map; common crop/threshold/components possible |
| YOLO IRSTD | invocation absent; metadata gives static B1/B4 `[B,3,512,640]`, opset 17, no NMS | final `[B,5,6720]` split into XY/WH/score | configs explicitly use split graph | Same semantic fields, **different tensor arity** (Torch/ordinary ONNX 1 vs CIX source 3); normalize first |
| YOLO IWT | same | same | same | same |

Preprocessing is not graph-embedded for any artifact. ALCNet expects already
normalized grayscale. YOLO ONNX/CIX expects already letterboxed RGB float32
NCHW in `[0,1]`; `/255`, channel conversion and letterbox are performed by
Ultralytics only when using high-level `.predict()`. Therefore `.pt`, `.onnx`
and `.cix` preprocessing is semantically the same only if CIXInference explicitly
reimplements the pinned Ultralytics letterbox contract, including scale/padding
metadata for reversing coordinates.

## F. Общий код

### Model architecture

- ALCNet Torch requires the BasicIRSTD architecture and state_dict; CIX does not.
- YOLO Torch requires Ultralytics' YOLO26 implementation; CIX does not.

### Preprocessing

One model-specific preprocessor can feed all backends:

- ALCNet: PIL-compatible grayscale, IRSTD mean/std, bottom/right pad, NCHW.
- Both YOLO weights: three-channel RGB model tensor, letterbox to 512x640,
  float32 `/255`, NCHW, retaining scale/pad. The current high-level runner hides
  these steps inside Ultralytics and passes BGR source; universal code must not
  preprocess twice.

### Postprocessing

- ALCNet can share crop, strict threshold, 8-connectivity, component bbox and
  max-pixel confidence across CIX/Torch.
- Both YOLO variants can share concatenation to canonical `[B,5,N]`, confidence
  filter, `cxcywh -> xyxy`, class-aware/agnostic NMS decision, reverse letterbox,
  clipping and result objects. There is only one class, so class-aware and
  class-agnostic NMS coincide today, but this should be explicit.

## G. Backend differences

| Concern | Torch CPU/CUDA | CIX/NPU |
|---|---|---|
| Load | state_dict + local class (ALCNet), or Ultralytics checkpoint (YOLO) | `NOE_Engine.EngineInfer(.cix)` |
| Device | `.to(torch.device)`; CUDA availability check/synchronization | vendor runtime/device lifecycle and flattened buffers |
| Batch | Torch architecture can run compatible batches; existing model APIs are not fixed by artifact | distinct static B1/B4 artifacts |
| Outputs | ALCNet Tensor; YOLO low-level one canonical tensor / high-level Results | ALCNet one flattened output; YOLO expected three split outputs |
| Numeric | float32 (YOLO checkpoint stores half but runtime may cast appropriately) | built with INT8 quantization, returned API dtype/layout must be validated on NPU |

**Один adapter для CIX + PyTorch:** да на уровне model-specific facade and
canonical pre/post contracts, но не как один backend executor. ALCNet needs two
load/run strategies but one output map. YOLO needs two runtime strategies and an
output-normalization step: concatenate CIX XY/WH/score before shared NMS. Calling
Ultralytics `.predict()` would duplicate shared postprocessing, so the universal
adapter must either (a) use low-level Torch raw inference, or (b) accept that the
Torch branch remains a separate high-level postprocessor. Option (a) is the
clean route but needs a pinned-version integration test not present here.

## H. Что передать в CIXInference

### ALCNet

- **Weight:** exact `ALCNet_400.pth.tar` matching SHA-256 from export report; it
  must first be recovered because it is not in Git.
- **Definitions:** `ASKCResNetFPN` and `AsymBiChaFuse`; `Net` is a thin selector
  and can be adapted, not necessarily copied wholesale.
- **Preprocessing:** PIL `I` conversion, fixed IRSTD mean/std, zero pad to 32,
  NCHW float32 and original size.
- **Raw/post contract:** one sigmoid probability map; crop, `>0.5`, 8-connected
  components, component max score and bbox.
- **CIX artifacts/config:** b1/b4 `.cix`; static shapes and output flattening.

### Both YOLO models

- **Weights:** both checked-in `.pt` files.
- **Dependency/architecture:** a version of `ultralytics` that contains YOLO26n,
  `DetectionModel` and all pickle-referenced modules. Do not copy BasicIRSTD
  `model/`; it is unrelated to YOLO.
- **Preprocessing contract:** exact Ultralytics inference letterbox for
  `[512,640]`, color conversion and `/255`; pin/version-test this behavior.
- **Raw output contract:** canonical decoded `[cx,cy,w,h,score]`; CIX three-output
  join; no graph NMS.
- **Postprocessing contract:** threshold 0.25 and IoU 0.7 defaults, NMS, reverse
  scale/pad, clip, class 0/name from checkpoint.
- **CIX artifacts/config:** B1/B4 model pairs and their split-ONNX configs as
  provenance evidence; NOE output order/shape test is still required.

### Minimal inference dependencies

| Dependency | Status | Why |
|---|---|---|
| `torch` | mandatory for both Torch backends | tensors, modules, checkpoint load |
| `torchvision` | mandatory for current ALCNet architecture | internal tensor Resize and BasicBlock import |
| `ultralytics` | mandatory only for YOLO `.pt` | pickle classes, loader, predictor/NMS |
| `numpy` | mandatory shared boundary | image/tensor buffers |
| `Pillow` | mandatory for byte-compatible current ALCNet grayscale | `convert("I")` |
| `opencv-python` | mandatory in current postprocessing | color conversion, connected components; likely NMS can remain Ultralytics/torch |
| `matplotlib`, `scikit-image` | imported by legacy `net.py/utils.py` but not intrinsically needed for ALCNet inference | remove from inference import path when adapting, rather than ship training/eval stack |
| `onnx`, `onnxruntime` | conversion/validation only | not required at runtime for Torch or CIX |
| `NOE_Engine` | CIX/NPU only | vendor inference API |
| training stack (`tqdm`, augmentation/loss/metrics, dataset tools) | not required | training/evaluation only |

## I. Неизвестные и риски

1. **ALCNet checkpoint отсутствует.** Its structure is validated by loaders and
   export reports, but this checkout cannot independently inspect keys/tensors or
   run Torch. Recover and hash-check the exact file before integration.
2. **No Python ML/CV packages are installed in this environment.** CPU/CUDA
   smoke runs were impossible. CUDA availability itself cannot be queried
   without PyTorch. A network-blocked transient `pip install onnx` was attempted
   but no project dependency was changed.
3. **Ultralytics exact version/date metadata could not be safely unpickled**
   without the dependency. The checkpoint requires YOLO26-era classes; pin the
   producing version from the original environment before relying on pickle
   compatibility/security. `torch.load`/Ultralytics load of untrusted `.pt` is
   code execution and must only use trusted artifacts.
4. **YOLO low-level Torch return container is unverified here.** The ONNX output
   semantics are proven by the committed graph transform, but the universal
   Torch raw adapter must be validated against ordinary ONNX on identical input.
5. **YOLO export command is absent.** Embedded ONNX metadata proves settings but
   not the exact command, tool version or calibration provenance.
6. **CIX YOLO output behavior is not covered by the existing Orange Pi runner.**
   It assumes one ALCNet output and `[B,1,640,512]`; three-output ordering,
   flattening and dtype need a device smoke test.
7. **Artifact/config naming differs.** CIX configs declare `_xy_wh.cix`, while
   committed binaries omit this suffix. Confirm these were only renamed and
   were built from the adjacent split ONNX.
8. **Dimension-order trap in ALCNet exporter defaults/docs.** Stored artifacts
   consistently use H=640/W=512; some symbolic constants/default prose suggest
   512/640. Treat tensor shape, not variable naming, as authoritative.
9. **YOLO class name string is not confirmed without unpickling.** The datasets
   are single-class and class id is 0, but CIXInference should read/pin the
   intended human-readable name rather than invent it.
10. **Quantization means numerical identity is not guaranteed.** Share semantic
    postprocessing and thresholds, while testing near-threshold behavior between
    float Torch and INT8 CIX.

## Рекомендуемая приёмочная проверка перед реализацией

1. Recover ALCNet checkpoint and run its existing `ModelRunner` on CPU and CUDA.
2. Build a clean environment with the producing Ultralytics version; load both
   `.pt`, record checkpoint version/names, and compare raw model tensor with
   ordinary ONNX for the same already-preprocessed `[1,3,512,640]` tensor.
3. On Orange Pi, record every YOLO CIX output's count, dtype and flattened size;
   reconstruct `[B,2,6720] + [B,2,6720] + [B,1,6720]` and compare detections.
4. Only then freeze CIXInference's canonical preprocessor and backend-neutral
   postprocessor contracts.

---

# Дополнение: inference-аудит ISTDU-Net / IRSTD-1K

Дата аудита: 2026-09-08. Исходный код, checkpoints и runtime не изменялись.

## Итоговый контракт

```text
checkpoint:
  Ожидаемый BasicIRSTD checkpoint:
    log/IRSTD-1K/ISTDU-Net_400.pth.tar
  В текущем checkout отсутствует.
  Формат по коду, который его создаёт и загружает:
    dict {
      "epoch": int,
      "state_dict": Net(...).state_dict(),
      "total_loss": list
    }
  Это не полный nn.Module и не голый state_dict.

architecture files:
  Минимальное вычислительное поддерево для прямого построения ISTDU_Net:
    model/ISTDUNet/model_ISTDUNet.py
    model/ISTDUNet/minet.py
    model/ISTDUNet/resnet2020.py
    model/ISTDUNet/splat.py
    model/ISTDUNet/eta.py
  Штатный BasicIRSTD loader дополнительно использует:
    net.py
    model/__init__.py
    loss.py (только потому, что Net создаёт SoftIoULoss)
    utils.py (из-за широких legacy imports; не нужен самой архитектуре)
  model/ISTDU-Net-main/** и его save_pth/ISTDU_Net/best.pth — отдельная
  upstream/demo-копия и не замена требуемому BasicIRSTD checkpoint.

loader:
  device = torch.device("cuda" if requested_cuda else "cpu")
  model = Net(model_name="ISTDU-Net", mode="test").to(device)
  checkpoint = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(checkpoint["state_dict"], strict=True)
  model.eval()
  with torch.inference_mode():
      raw = model(input_tensor.to(device))

preprocessing:
  Входное изображение: PIL convert("I"), то есть одноканальное grayscale.
  NumPy dtype перед нормализацией: float32.
  IRSTD-1K normalization:
    (pixel - 87.4661865234375) / 39.71953201293945
  Внешнего resize в штатном BasicIRSTD path нет.
  PadImg дополняет нулями только снизу и справа до кратности 32.
  Tensor: float32, NCHW, [B, 1, Hpad, Wpad].
  Проверенный экспортный контракт: [1, 1, 640, 512].

raw output:
  Один torch.Tensor, не tuple/list.
  Shape: [B, 1, Hpad, Wpad], совпадает с пространственным размером input.
  Для экспортного контракта: [1, 1, 640, 512].
  funOutput() возвращает torch.sigmoid(self.headSeg(x)); sigmoid уже внутри.
  Математический диапазон: [0, 1]. Дополнительный sigmoid не нужен.
  headDet создан в объекте и присутствует в state_dict, но текущий forward его
  не возвращает и в inference не использует.

postprocessing:
  Обрезать padding до исходных H,W.
  binary_mask = probability_map > 0.5 (строгое сравнение).
  8-connected components через OpenCV.
  Для каждой foreground-компоненты:
    bbox = [x, y, x + width, y + height]
    centroid = connectedComponentsWithStats centroid
    area = число пикселей компоненты
    confidence = max(probability_map[labels == component_label])
  NMS, objectness, class scores и bbox decode отсутствуют.

dependencies:
  Минимальная архитектура: torch.
  Штатный preprocessing: numpy + Pillow; torchvision не требуется самой
  ISTDU-Net, но импортируется текущими net.py/utils.py/model/__init__.py.
  Текущий общий postprocessing: opencv-python.
  CPU и CUDA используют один код; CUDA требует CUDA-enabled torch/driver.
  matplotlib, scikit-image, ONNX и ONNX Runtime не нужны для runtime inference.

recommended CIXInference adapter:
  Использовать существующий segmentation/probability-map adapter ALCNet, а не
  отдельный IstdUNetAdapter, если adapter получает model loader/factory и
  preprocessing policy через конфигурацию.
  Общими являются output contract, crop, threshold 0.5, connected components,
  bbox и component confidence.
  Различаются только Torch architecture/weights и внутренняя сеть; normalization
  сейчас тоже совпадает, потому что обе модели обучены на IRSTD-1K.
  Отдельный IstdUNetAdapter оправдан только если текущий ALCNet adapter жёстко
  зашивает класс ALCNet или его export-only Resize workaround.
```

## Подтверждение архитектуры и выхода

`net.Net` для имени `ISTDU-Net` создаёт `ISTDU_Net` из
`model/ISTDUNet/model_ISTDUNet.py`; wrapper только делегирует `forward`. Сам
`ISTDU_Net` наследует `miNet`, поэтому последовательность фактического forward:

```text
miNet.forward
  -> ISTDU_Net.funIndividual: Down (stem + ResNetCt)
  -> funPallet/funConbine/funEncode: identity
  -> funDecode: EDN external attention + UPCt bilinear upsampling/skips
  -> funOutput: sigmoid(headSeg)
  -> one probability-map Tensor
```

В `model_ISTDUNet.py` одновременно объявлен `headDet`, но строка возврата
детектора закомментирована. Это важно не путать с upstream-копией
`model/ISTDU-Net-main/model/ctNet/ctNet.py`: там `forward` возвращает два
sigmoid tensors `(headDet, headSeg)`. BasicIRSTD импортирует не эту копию, а
модифицированный `model/ISTDUNet/model_ISTDUNet.py` с одним segmentation output.

## Checkpoint: что подтверждено и что отсутствует

Точный путь формируется training/test/inference кодом как
`log/<dataset>/<model>_400.pth.tar`; для этой модели и набора это
`log/IRSTD-1K/ISTDU-Net_400.pth.tar`. Export script содержит тот же абсолютный
от корня default. Имя также присутствует в сохранённом test log, где приведены
метрики IRSTD-1K. Сам файл не находится ни в рабочем дереве, ни в истории Git,
поэтому фактические ключи, tensor shapes и SHA-256 именно этого файла повторно
проинспектировать нельзя.

Формат всё же однозначно задаётся `train.py`: `save_checkpoint` вызывает
`torch.save()` для dict из `epoch`, `state_dict=net.module.state_dict()` и
`total_loss`; все BasicIRSTD loaders берут именно `checkpoint["state_dict"]`.
`ModelRunner` дополнительно валидирует dict и выполняет strict load.

В репозитории есть
`model/ISTDU-Net-main/save_pth/ISTDU_Net/best.pth` (SHA-256
`25e237f84c3951050acc1358c39960e1caffe8aff7c7facb9c3d20c8d8ee916c`), но его
standalone `detect.py` загружает файл как голый state_dict в `DataParallel` и
ожидает upstream two-output model. Подменять им `ISTDU-Net_400.pth.tar` без
отдельной migration/key/output проверки нельзя.

## Batch 1/2/4

Архитектурный код не фиксирует batch и все операции сохраняют batch dimension;
в eval mode нет batch-dependent Python logic. Поэтому PyTorch inference
архитектурно допускает B=1, B=2 и B=4 при одинаковых H/W и достаточной памяти.
Однако в этом окружении это не было исполнено из-за отсутствия target checkpoint
и PyTorch.

Текущий ONNX exporter и CIX config фиксированы только на B=1. Наличие PyTorch
batch support не доказывает поддержку B=2/B=4 существующим ONNX/CIX artifact:
для них нужны отдельные статические exports/builds либо подтверждённый dynamic
контракт. В checkout нет ни одного ISTDU-Net ONNX/CIX artifact.

## Конвертация CIX и причина отсутствия результата

Проект содержит законченный по коду pipeline:

```text
ISTDU-Net_400.pth.tar
  -> ModelRunner CPU + штатный preprocessing [1,1,640,512]
  -> torch.onnx.export, opset 17, constant folding, static axes,
     input="input", output="probability_map"
  -> ONNX checker + ONNX Runtime numerical/threshold/component comparison
  -> CixBuilder config, INT8 calibration, X2_1204MP3
  -> istdunet_irstd1k.cix
```

Но результат pipeline не закоммичен: каталог
`iwt_tools/models/istdunet_irstd1k`, ONNX, export report и CIX отсутствуют во
всей доступной истории Git. Также нет сохранённого stdout/stderr CixBuilder или
текста с названием неподдержанного оператора. Поэтому утверждать, что
конвертация упала из-за конкретной операции (например Softmax, Div,
interpolation или reshape), нельзя: такая причина **не зафиксирована в
проекте**. Единственные подтверждённые блокеры воспроизведения в этом checkout —
отсутствующие target checkpoint, тестовый внешний кадр, calibration data и
ML/CIX toolchain. Наличие config и deploy-примера само по себе не подтверждает
успешный build.

## Решение по adapter

ISTDU-Net и ALCNet имеют один backend-neutral semantic contract:

```text
normalized grayscale NCHW
  -> one already-sigmoid probability map of input spatial size
  -> crop original size
  -> strict > 0.5
  -> 8-connected components
  -> bbox + centroid + area + max-pixel confidence
```

Следовательно, отдельный postprocessor и отдельный публичный
`IstdUNetAdapter` не нужны. Рекомендуется общий segmentation adapter с
инъецируемыми `model_factory`, checkpoint loader и preprocessing config. При
этом ALCNet-specific export Resize workaround нельзя помещать в общий runtime
adapter: он относится только к конвертации ALCNet. Это рекомендация для
`CIXInference`; в BasicIRSTD ничего не переносилось и не рефакторилось.
