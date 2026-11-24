"""160 ms acoustic window features shared by PartialSpoof training and inference."""
import io
from math import gcd

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

STEP=2560


def windows(source):
    with sf.SoundFile(io.BytesIO(source) if isinstance(source,bytes) else source) as audio:
        waveform=audio.read(min(len(audio),audio.samplerate*120),dtype='float32',always_2d=True).mean(1)
        if audio.samplerate!=16000:
            factor=gcd(audio.samplerate,16000)
            waveform=resample_poly(waveform,16000//factor,audio.samplerate//factor)
    if not waveform.size:raise ValueError('Empty audio recording.')
    duration=len(waveform)/16000
    waveform=np.pad(waveform,(0,(-len(waveform))%STEP))
    chunks=waveform.reshape(-1,STEP)
    short=chunks.reshape(-1,8,320)
    spectrum=np.abs(np.fft.rfft(short*np.hanning(320),axis=-1))[:,:,1:]
    bands=np.log1p(spectrum.reshape(-1,8,32,5).mean(-1))
    mean=bands.mean(1)
    context=np.abs(mean-np.vstack((mean[:1],mean[:-1])))
    features=np.hstack((mean,bands.std(1),context,np.abs(np.diff(bands,axis=1)).mean(1)))
    return features.astype(np.float32),duration


def clip_score(scores):
    return float(np.sort(scores)[-min(3,len(scores)):].mean())


def intervals(scores,threshold,duration):
    flags=np.pad(scores>=threshold,(1,1))
    changes=np.flatnonzero(flags[1:]!=flags[:-1]).reshape(-1,2)
    return [[round(start*.16,2),round(min(end*.16,duration),3)] for start,end in changes]
