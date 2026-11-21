"""Install the optional pretrained English transcription model on the project drive."""
import hashlib
from pathlib import Path
import zipfile

from ranged_download import transfer

ROOT = Path(__file__).resolve().parents[1]
NAME = "vosk-model-small-en-us-0.15"
folder = ROOT / "data/public/asr"
archive = folder / (NAME + ".zip")
transfer(f"https://alphacephei.com/vosk/models/{NAME}.zip", archive)
expected = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError("The Vosk archive differs from the verified release.")
with zipfile.ZipFile(archive) as source:
    for member in source.infolist():
        if not (folder / member.filename).resolve().is_relative_to((folder / NAME).resolve()):
            raise ValueError("Unexpected model archive member")
    source.extractall(folder)
print("Installed", NAME, "archive sha256", hashlib.sha256(archive.read_bytes()).hexdigest())
