"""A development audio baseline when only official LA training data is acquired.

Never presents these internal metrics as official ASVspoof eval performance.
Original speaker-disjoint challenge validation and evaluation remain untouched.
"""
from argparse import Namespace
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.train_public import splits
from training.train_media import train

def main():
    source=ROOT/'data/prepared/asvspoof/train.jsonl'
    while not source.exists():
        print('Waiting for completed training-partition preprocessing',flush=True);time.sleep(30)
    rows=[json.loads(l) for l in source.read_text().splitlines()]
    tr,va,te=splits([r['group'] for r in rows])
    folder=ROOT/'data/prepared/asvspoof-internal';folder.mkdir(parents=True,exist_ok=True)
    for name,indices in [('train',tr),('val',va),('test',te)]:
        (folder/f'{name}.jsonl').write_text(''.join(json.dumps({**rows[i],'split':name})+'\n' for i in indices),encoding='utf-8')
    train(Namespace(task='audio',data=str(folder),dataset_name='ASVspoof 2019 LA training partition only (internal speaker split)',
        epochs=10,batch=64,lr=.0005,evaluation_scope='internal-training-partition-only',
        limitations='Development benchmark only: internal speaker-disjoint 80/10/10 split of the 25,380 official training utterances. Only six known training attacks; official development/evaluation and unseen-attack testing are NOT completed. First six seconds only. Cannot identify cloning, speaker identity, or splices.'))

if __name__=='__main__':main()
