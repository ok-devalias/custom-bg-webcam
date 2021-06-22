import camera
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


def process_image(height, width, fps, image_filename=BG_IMG):
    path = Path(Path.cwd() / image_filename)
    img = Image.open(path)
    seq = ImageSequence.Iterator(img)
    resized_seq = [frame.resize((width, height)) for frame in seq]
    mirrored_seq = [ImageOps.mirror(frame) for frame in resized_seq]
    return mirrored_seq

def activate_cam(height, width, fps, device=0, api=cv2.CAP_DSHOW):
    return camera.Camera(height, width, fps, device, api)


def rgb_to_rgba(frame, height, width):
    return np.dstack(
        (frame, np.zeros((height, width), dtype=np.uint8)+255))


def loop_to_cam(image_sequence, webcam):
    with pyvirtualcam.Camera(webcam.width, webcam.height, webcam.fps,
     fmt=PixelFormat.RGBA, device=CAM_DEVICE) as cam:
        print(f'cam: {cam.device}')
        frame = np.zeros((cam.height, cam.width, 4), np.uint8)
        while True:
            try:
                for f in image_sequence:
                    c_frame = webcam.get_frame(toRgb=True)
                    mask = get_mask(c_frame)
                    inverse_mask = 1-mask
                    rgba = rgb_to_rgba(c_frame, cam.height, cam.width)
                    frame[:] = 0
                    bg_frame = np.array(f)
                    for c in range(frame.shape[2]):
                        frame[:,:,c] = rgba[:,:,c]*mask + bg_frame[:,:,c]*inverse_mask
                    cam.send(frame)
                    cam.sleep_until_next_frame()
            except KeyboardInterrupt:
                break
