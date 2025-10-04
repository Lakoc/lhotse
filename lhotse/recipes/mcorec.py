import itertools
import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.recipes.utils import TimeFormatConverter, normalize_text_chime6
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations, resumable_download


def download_mcorec(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    hf_token: str = "",
) -> Path:
    base_url = "https://huggingface.co/datasets/MCoRecChallenge/MCoRec/resolve/main"
    files_to_download = (
        "dev_without_central_videos.zip",
        "dev_only_central_videos.zip",
        "train_without_central_videos.zip",
        "train_only_central_videos.zip",
    )
    auth_header = {"Authorization": f"Bearer {hf_token}"}
    target_dir = Path(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    for zip_name in files_to_download:
        zip_path = target_dir / zip_name
        resumable_download(
            f"{base_url}/{zip_name}",
            filename=zip_path,
            force_download=force_download,
            additional_headers=auth_header,
        )

        # Remove partial unpacked files, if any, and unpack everything.
        with zipfile.ZipFile(zip_path) as zip_f:
            zip_f.extractall(path=target_dir)

    return Path(target_dir)


def prepare_mcorec(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dataset_parts: Optional[Union[str, Sequence[str]]] = "all",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    try:
        import webvtt
    except ImportError:
        raise ImportError(
            "webvtt is not installed, please, run `pip install webvtt-py`."
        )

    # available_parts = set(("train", "dev"))
    available_parts = set(("dev",))
    parts_to_process = []

    if isinstance(dataset_parts, list) or isinstance(dataset_parts, tuple):
        if len(dataset_parts) == 1:
            dataset_parts = dataset_parts[0]
        else:
            for part in dataset_parts:
                if part not in available_parts:
                    raise ValueError(
                        f"{part} is not a valid dataset part. Choose one of: {available_parts}."
                    )
                parts_to_process.append(part)

    # Validate parts
    if isinstance(dataset_parts, str):
        if dataset_parts == "all":
            parts_to_process = list(available_parts)
        elif dataset_parts not in available_parts:
            raise ValueError(
                f"{dataset_parts} is not a valid dataset part. Choose one of: {available_parts}."
            )
        else:
            parts_to_process = [dataset_parts]

    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    manifests = defaultdict(dict)

    for part in parts_to_process:
        logging.info(f"Processing {part}")

        part_dir = corpus_dir / part
        if not os.path.exists(part_dir):
            raise FileNotFoundError(f"{part} directory {part_dir} does not exist.")

        recordings = []
        supervisions = []

        sessions = os.listdir(part_dir)
        for session in tqdm(sessions, desc=f"Processing {part} sessions"):
            session_dir = part_dir / session
            video_path = session_dir / "central_video.mp4"
            audio_path = session_dir / "central_audio.wav"
            labels_path = session_dir / "labels"

            _convert_vid_to_single_ch_audio(video_path, audio_path)
            recording = Recording.from_file(audio_path)
            recording.id = f"{part}_{session}_{recording.id}"
            recordings.append(recording)

            for speaker_labels_file in filter(
                lambda s: ".vtt" in s, os.listdir(labels_path)
            ):
                speaker_name = speaker_labels_file.split(".")[0]
                for i, caption in enumerate(
                    webvtt.read(labels_path / speaker_labels_file)
                ):
                    st = _convert_time_str_to_float(caption.start)
                    et = _convert_time_str_to_float(caption.end)
                    supervisions.append(
                        SupervisionSegment(
                            id=f"{recording.id}_{speaker_name}_{i}",
                            recording_id=recording.id,
                            start=st,
                            duration=et - st,
                            text=caption.text,
                            channel=[0],
                            speaker=speaker_name,
                            language="english",
                        )
                    )

        recordings = RecordingSet.from_recordings(recordings)
        supervisions = SupervisionSet.from_segments(supervisions)

        recording_set, supervision_set = fix_manifests(
            recordings=recordings, supervisions=supervisions
        )

        # Fix manifests
        validate_recordings_and_supervisions(recording_set, supervision_set)

        recording_set.to_file(output_dir / f"mcorec_recordings_{part}.jsonl.gz")
        supervision_set.to_file(output_dir / f"mcorec_supervisions_{part}.jsonl.gz")

        manifests[part]["recordings"] = recording_set
        manifests[part]["supervisions"] = supervision_set

    return manifests


def _convert_vid_to_single_ch_audio(
    video_path, output_path, skip_existing: bool = True
) -> None:
    if os.path.exists(output_path) and skip_existing:
        return

    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _convert_time_str_to_float(time_str: str) -> float:
    h, m, s = time_str.split(":")
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds
