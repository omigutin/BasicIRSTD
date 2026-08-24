"""Выбирает кадры из видео через заданный временной интервал.

Запускать из корня BasicIRSTD::

    python iwt_tools/scripts/data/sample_video_frames.py \
        --input path/to/videos \
        --output path/to/video_benchmark \
        --interval-sec 3
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".wmv",
}


@dataclass(frozen=True)
class ExtractedFrame:
    video: str
    source_path: str
    image: str
    frame_index: int
    timestamp_sec: float


def _safe_name(value: str) -> str:
    """Преобразует имя в безопасный фрагмент имени файла."""
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE)
    return value.strip("._") or "video"


def _collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    """Возвращает список видеофайлов из файла или каталога."""
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Неподдерживаемое расширение видео: {input_path.suffix}")
        return [input_path.resolve()]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Входной путь не найден: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    videos = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def _extract_from_video(
    video_path: Path,
    frames_dir: Path,
    interval_sec: float,
) -> list[ExtractedFrame]:
    """Последовательно читает видео и сохраняет кадр через заданный интервал."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError(f"Не удалось определить FPS: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        interval_frames = max(1, round(interval_sec * fps))

        print(
            f"\n{video_path.name}\n"
            f"  FPS: {fps:.6f}\n"
            f"  Кадров: {frame_count if frame_count > 0 else 'unknown'}\n"
            f"  Шаг: {interval_frames} кадров (~{interval_frames / fps:.3f} сек)"
        )

        stem = _safe_name(video_path.stem)
        records: list[ExtractedFrame] = []
        frame_index = 0
        next_frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index >= next_frame_index:
                timestamp_sec = frame_index / fps
                timestamp_ms = round(timestamp_sec * 1000)

                filename = (
                    f"{stem}"
                    f"__f{frame_index:09d}"
                    f"__t{timestamp_ms:010d}ms.png"
                )
                output_path = frames_dir / filename

                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Не удалось сохранить кадр: {output_path}")

                records.append(
                    ExtractedFrame(
                        video=video_path.name,
                        source_path=str(video_path),
                        image=str(Path("frames") / filename),
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                    )
                )
                next_frame_index += interval_frames

            frame_index += 1

        print(f"  Сохранено кадров: {len(records)}")
        return records
    finally:
        capture.release()


def _write_manifest(records: list[ExtractedFrame], output_path: Path) -> None:
    """Сохраняет соответствие изображений исходным видео и временным меткам."""
    manifest_path = output_path / "frames_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["video", "source_path", "image", "frame_index", "timestamp_sec"])
        for record in records:
            writer.writerow([
                record.video,
                record.source_path,
                record.image,
                record.frame_index,
                f"{record.timestamp_sec:.6f}",
            ])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Нарезает видео на PNG-кадры через заданный временной интервал "
            "для последующей разметки в Roboflow."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Путь к одному видео или каталогу с видео.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Каталог для сохранения кадров и frames_manifest.csv.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=3.0,
        help="Интервал между кадрами в секундах. По умолчанию: 3.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Искать видео рекурсивно во вложенных каталогах.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    if args.interval_sec <= 0:
        raise ValueError("--interval-sec должен быть больше 0.")

    videos = _collect_videos(args.input, recursive=args.recursive)
    if not videos:
        raise RuntimeError(f"Видео не найдены: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Найдено видео: {len(videos)}")
    print(f"Интервал: {args.interval_sec} сек")
    print(f"Результат: {args.output.resolve()}")

    all_records: list[ExtractedFrame] = []
    for video_path in videos:
        all_records.extend(
            _extract_from_video(
                video_path=video_path,
                frames_dir=frames_dir,
                interval_sec=args.interval_sec,
            )
        )

    _write_manifest(all_records, args.output)

    print("\nГотово.")
    print(f"Всего сохранено кадров: {len(all_records)}")
    print(f"Кадры: {frames_dir.resolve()}")
    print(f"Manifest: {(args.output / 'frames_manifest.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
