from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from api.pipeline_runner import PipelineRunner
from models_data.segment import Segment
from pipeline.p08_audio_alignment import AudioAligner


def test_audio_aligner_creates_output_files(tmp_path):
    audio_path = tmp_path / "input.wav"
    sr = 16000
    duration = 0.2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(audio_path, audio, sr)

    segment = Segment(id=1, generated_audio=str(audio_path))
    aligner = AudioAligner()

    result = aligner.process([segment], str(tmp_path / "stage8"))

    assert len(result) == 1
    assert Path(segment.aligned_audio).exists()
    assert (tmp_path / "stage8" / "segment_1.wav").exists()
    assert (tmp_path / "stage8" / "merged_audio.wav").exists()


def test_pipeline_runner_persists_stage_outputs(tmp_path):
    runner = PipelineRunner()
    runner.settings = SimpleNamespace(output_dir=tmp_path)

    output_file = tmp_path / "demo.mp4"
    output_file.write_bytes(b"demo")

    stage_dir = runner._persist_stage_artifacts(
        9,
        "Lip Sync",
        [str(output_file)],
        [str(output_file)],
        {"status": "ok"},
    )

    assert stage_dir.exists()
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "data.json").exists()
    assert (stage_dir / output_file.name).exists()
