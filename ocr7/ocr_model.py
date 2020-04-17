# pylint: disable=too-few-public-methods
import numpy as np

from . import detection, recognition, utils


class Pipeline:
    def __init__(self, detector=None, recognizer=None, scale=2, max_size=2048):
        if detector is None:
            detector = detection.Detector()
        if recognizer is None:
            recognizer = recognition.Recognizer()
        self.scale = scale
        self.detector = detector
        self.recognizer = recognizer
        self.max_size = max_size

    def recognize(self, images, detection_kwargs=None, recognition_kwargs=None):
        # Make sure we have an image array to start with.
        if not isinstance(images, np.ndarray):
            images = [utils.read(image) for image in images]
        # This turns images into (image, scale) tuples temporarily
        images = [
            utils.resize_image(image, max_scale=self.scale, max_size=self.max_size)
            for image in images
        ]
        max_height, max_width = np.array([image.shape[:2] for image, scale in images]).max(axis=0)
        scales = [scale for _, scale in images]
        images = np.array(
            [utils.pad(image, width=max_width, height=max_height) for image, _ in images])
        if detection_kwargs is None:
            detection_kwargs = {}
        if recognition_kwargs is None:
            recognition_kwargs = {}
        box_groups = self.detector.detect(images=images, **detection_kwargs)
        prediction_groups = self.recognizer.recognize_from_boxes(images=images,
                                                                 box_groups=box_groups,
                                                                 **recognition_kwargs)
        box_groups = [
            utils.adjust_boxes(boxes=boxes, boxes_format='boxes', scale=1 /
                                                                        scale) if scale != 1 else boxes
            for boxes, scale in zip(box_groups, scales)
        ]
        return [
            list(zip(predictions, boxes))
            for predictions, boxes in zip(prediction_groups, box_groups)
        ]
