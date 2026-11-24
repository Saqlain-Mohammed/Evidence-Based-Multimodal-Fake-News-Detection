"""Frame discontinuity features for a source-separated video deletion benchmark."""
import cv2
import numpy as np
from scipy.ndimage import median_filter


def transitions(path):
    capture=cv2.VideoCapture(str(path))
    try:
        fps=capture.get(cv2.CAP_PROP_FPS)
        if not 0<fps<=240:raise ValueError("Invalid video frame rate")
        previous=None
        features=[]
        for index in range(min(3600,int(fps*120))):
            ok,frame=capture.read()
            if not ok:break
            gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(48,32),interpolation=cv2.INTER_AREA).astype(np.float32)/255
            if previous is not None:
                difference=np.abs(gray-previous)
                edges=np.abs(cv2.Laplacian(gray,cv2.CV_32F)-cv2.Laplacian(previous,cv2.CV_32F))
                tiles=difference.reshape(4,8,6,8).mean(axis=(1,3))
                features.append([difference.mean(),difference.std(),np.quantile(difference,.95),tiles.max(),
                                 np.median(tiles),edges.mean(),abs(gray.mean()-previous.mean())])
            previous=gray
        if len(features)<2:raise ValueError("Video too short for temporal edit analysis")
        x=np.asarray(features,dtype=np.float32)
        context=median_filter(x,size=(11,1),mode="nearest")
        return np.hstack((x,x-context,x/(.015+context))),fps
    finally:capture.release()


def candidates(scores,threshold,fps):
    order=np.argsort(scores)[::-1]
    selected=[]
    for index in order:
        if scores[index]<threshold or len(selected)>=8:break
        if all(abs(index-other)>3 for other in selected):selected.append(index)
    return [round((int(index)+1)/fps,3) for index in sorted(selected)]
