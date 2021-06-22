import cv2

class Camera:
    """Camera object for setting up and interacting with a webcam."""

    def __init__(self, height, width, fps, device=0, api=None):
        self.height = height
        self.width = width
        self.fps = fps
        if not api:
            self.capture = cv2.VideoCapture(device)
        else:
            self.capture = cv2.VideoCapture(device, api)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

    def get_frame(self, toRgb=False):
        _, frame = self.capture.read()
        if toRgb:
            frame = self._bgr_to_rgb(frame)
        return frame

    def _bgr_to_rgb(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
