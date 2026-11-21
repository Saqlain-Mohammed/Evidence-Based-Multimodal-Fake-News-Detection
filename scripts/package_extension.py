"""Package Chromium/Edge and Firefox manifests without installing anything."""
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'extension';out=ROOT/'dist';out.mkdir(exist_ok=True)
for browser in ['chromium','firefox']:
    manifest=json.loads((folder/'manifest.json').read_text())
    if browser=='firefox':
        manifest['background']={'scripts':['background.js']}
        manifest['browser_specific_settings']={'gecko':{'id':'veritas@local.research','strict_min_version':'128.0'}}
    with zipfile.ZipFile(out/f'veritas-{browser}.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
        for file in folder.iterdir():
            if file.is_file() and file.name!='manifest.json':z.write(file,file.name)
    print(out/f'veritas-{browser}.zip')
