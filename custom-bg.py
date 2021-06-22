import cv2
import numpy as np
import pyvirtualcam
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence
from pyvirtualcam import PixelFormat

CAM_DEVICE = 'Unity Video Capture'
BG_IMG = 'pixelart.webp'

bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16
))

def get_mask(frame):
    result = bodypix_model.predict_single(np.array(frame))
    mask = np.array(result.get_mask(threshold=0.75, dtype=np.uint8))
    mask = mask.reshape((frame.shape[0], frame.shape[1]))
    return mask

def loop_to_cam():
    path = Path(Path.cwd() / BG_IMG)
    height, width = 1080, 1920
    fps = 60
    img = Image.open(path)
    seq = ImageSequence.Iterator(img)
    resized_seq = [frame.resize((width, height)) for frame in seq]
    mirrored_seq = [ImageOps.mirror(frame) for frame in resized_seq]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.RGBA, device=CAM_DEVICE) as cam:
        print(f'cam: {cam.device}')
        frame = np.zeros((cam.height, cam.width, 4), np.uint8)
        while True:
            try:
                for f in mirrored_seq:
                    _, c_frame = cap.read()
                    mask = get_mask(c_frame)
                    inverse_mask = 1-mask
                    c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
                    rgba = np.dstack(
                        (c_frame, np.zeros((height, width), dtype=np.uint8)+255)
                    )
                    frame[:] = 0
                    bg_frame = np.array(f)
                    for c in range(frame.shape[2]):
                        frame[:,:,c] = rgba[:,:,c]*mask + bg_frame[:,:,c]*inverse_mask
                    cam.send(frame)
                    cam.sleep_until_next_frame()
            except KeyboardInterrupt:
                break
