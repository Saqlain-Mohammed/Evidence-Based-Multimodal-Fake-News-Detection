"""Offline English speech recognition for the first two minutes of a recording."""
from functools import lru_cache
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL = Path(__file__).resolve().parents[1] / "data/public/asr/vosk-model-small-en-us-0.15"


@lru_cache(maxsize=1)
def recognizer_model():
    from vosk import Model
    return Model(str(MODEL))


def transcript(path):
    if not (MODEL / "am/final.mdl").is_file():
        return {"status": "English speech model unavailable", "text": ""}
    try:
        from vosk import KaldiRecognizer
    except (ImportError, OSError):
        return {"status": "Speech runtime unavailable", "text": ""}
    recognizer = KaldiRecognizer(recognizer_model(), 16000)
    phrases = []
    with sf.SoundFile(path) as audio:
        maximum = min(len(audio), audio.samplerate * 120)
        divisor = gcd(16000, audio.samplerate)
        while audio.tell() < maximum:
            samples = audio.read(min(audio.samplerate, maximum - audio.tell()), dtype="float32", always_2d=True).mean(axis=1)
            if audio.samplerate != 16000:
                samples = resample_poly(samples, 16000 // divisor, audio.samplerate // divisor)
            sound = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            if recognizer.AcceptWaveform(sound):
                phrases.append(json.loads(recognizer.Result())["text"])
        duration = maximum / audio.samplerate
    phrases.append(json.loads(recognizer.FinalResult())["text"])
    text = " ".join(filter(None, phrases))
    return {"status": "Transcribed" if text else "No English speech recognized", "text": text,
            "language": "en-US", "seconds": duration, "engine": "Vosk small English 0.15 (pretrained)"}


def decode_phone_recording(path):
    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe
    target = path.with_suffix(".wav")
    args = [get_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-protocol_whitelist", "file",
            "-f", "mov", "-i", str(path), "-map", "0:a:0", "-vn", "-t", "120",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target)]
    try:
        subprocess.run(args, check=True, capture_output=True, timeout=45)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Unable to decode the phone recording.") from exc
    return target
