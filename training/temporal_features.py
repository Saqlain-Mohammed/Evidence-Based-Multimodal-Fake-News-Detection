"""Clip descriptors capturing spatial residuals and changes across sampled frames."""
import cv2
import numpy as np


def describe(frames):
    rgb=frames.permute(0,2,3,1).numpy()
    gray=np.stack([cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) for frame in rgb])
    residual=np.stack([np.abs(cv2.Laplacian(frame,cv2.CV_32F)) for frame in gray])
    delta=np.abs(np.diff(gray,axis=0))
    values=[]
    for cube in (gray,residual,delta):
        for region in (cube,cube[:,16:48,16:48]):
            for series in (region.mean(axis=(1,2)),region.std(axis=(1,2))):
                values.extend(np.percentile(series,[10,50,90]).tolist())
                values.extend([float(series.mean()),float(series.std())])
    return np.asarray(values,dtype=np.float32)


def temporal_changes(frames,times):
    # Explain sampled visual changes without claiming to identify a forged frame.
    delta=(frames[1:]-frames[:-1]).abs().mean(dim=(1,2,3)).numpy()
    return [{'from_seconds':times[i],'to_seconds':times[i+1],'mean_rgb_change':round(float(delta[i]),4)}
            for i in np.argsort(delta)[::-1][:3]]
