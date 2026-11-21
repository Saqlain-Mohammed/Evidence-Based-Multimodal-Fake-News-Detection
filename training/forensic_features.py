"""Summarize native-resolution compression and residual traces for image classification."""
import io

import cv2
import numpy as np
from PIL import Image


def features(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image.thumbnail((1024,1024))
        image=image.convert("RGB")
        original=np.asarray(image).astype(np.float32)
    h,w=original.shape[:2]
    if min(h,w)<16:
        original=cv2.resize(original,(max(w,16),max(h,16)));h,w=original.shape[:2]
    values=[]
    def describe(array):
        absolute=np.abs(array)
        values.extend([float(absolute.mean()),float(absolute.std()),*np.percentile(absolute,[25,50,75,90,98]).tolist()])
        tile=cv2.resize(absolute,(8,8),interpolation=cv2.INTER_AREA)
        values.extend(np.percentile(tile,[5,25,50,75,95]).tolist())
    for quality in (80,95):
        _,encoded=cv2.imencode(".jpg",original.astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,quality])
        difference=original-cv2.imdecode(encoded,cv2.IMREAD_COLOR).astype(np.float32)
        for channel in range(3):describe(difference[:,:,channel])
    gray=cv2.cvtColor(original,cv2.COLOR_RGB2GRAY)
    for residual in (cv2.Laplacian(gray,cv2.CV_32F), gray-cv2.GaussianBlur(gray,(3,3),0)):
        describe(residual)
    for axis in (0,1):
        gradient=np.moveaxis(np.abs(np.diff(gray,axis=axis)),axis,0)
        energies=np.array([gradient[offset::8].mean() for offset in range(8)])
        values.extend((energies/(energies.mean()+1e-6)).tolist())
    return np.asarray(values,dtype=np.float32)
