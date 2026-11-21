"""Local color and residual features for supervised CASIA tamper masks."""
import io
import cv2
import numpy as np
from PIL import Image


def pixels(raw):
    with Image.open(io.BytesIO(raw)) as image:
        rgb = np.asarray(image.convert("RGB").resize((64,64)), dtype=np.float32) / 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    channels = [rgb, cv2.Laplacian(gray,cv2.CV_32F)[:,:,None]]
    for size in (3, 7, 15):
        mean = cv2.blur(rgb, (size,size))
        variance = np.maximum(0,cv2.blur(rgb**2,(size,size))-mean**2)
        channels.extend((rgb-mean,np.sqrt(variance)))
    return np.concatenate(channels,axis=2).reshape(4096,-1)
