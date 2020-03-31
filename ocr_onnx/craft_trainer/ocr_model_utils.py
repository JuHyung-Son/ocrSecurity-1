import os
import sys

import io
import cv2
import json
import typing
import random

import numpy as np

def read_tool(filepath_or_buffer: typing.Union[str, io.BytesIO]):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file, a URL, or any object
            with a `read` method (such as `io.BytesIO`)
    """
    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer
    if hasattr(filepath_or_buffer, 'read'):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(filepath_or_buffer, str):
        if validators.url(filepath_or_buffer):
            return read(urllib.request.urlopen(filepath_or_buffer))
        assert os.path.isfile(filepath_or_buffer), \
            'Could not find image at path: ' + filepath_or_buffer
        image = cv2.imread(filepath_or_buffer)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_gaussian_heatmap(size=512, distanceRatio=3.3):
    dis_ratio = random.randint(32, 35)
    dis_ratio /= 10.0
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    x, y = np.meshgrid(v, v)
    g = np.sqrt(x**2 + y**2)
    #g *= distanceRatio / (size / 2)
    g *= dis_ratio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype('uint8')

def compute_input(image):
    # should be RGB order
    image = image.astype('float32')
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    image -= mean * 255
    image /= variance * 255
    return image

def compute_maps(image_height, image_width, annotation):
    assert image_height % 2 == 0, 'Height must be an even number'
    assert image_width % 2 == 0, 'Width must be an even number'

    textmap = np.zeros((image_height // 2, image_width // 2)).astype('float32')
    linkmap = np.zeros((image_height // 2, image_width // 2)).astype('float32')
  
    orientation = 'horizontal'
    previous_link_points = None

    txt_list = annotation["annotations"][0]["text"]
    bbox_list = annotation["annotations"][0]["bbox"]

    for idx in range(len(txt_list)):
        txt = txt_list[idx]["contents"]
        box = bbox_list[idx]["contents"]

        x1, y1 = box[0] - (box[2] / 2), box[1] - (box[3] / 2) # left top
        x2, y2 = box[0] + (box[2] / 2), box[1] - (box[3] / 2) # right top
        x3, y3 = box[0] + (box[2] / 2), box[1] + (box[3] / 2) # right bottom
        x4, y4 = box[0] - (box[2] / 2), box[1] + (box[3] / 2) # left bottom

        width = int(x2 - x1)
        height = int(y3 - y1)

        heatmap = get_gaussian_heatmap(size=512)
        heatmap = cv2.resize(heatmap, (width, height)) # w, h

        src = np.array([[0, 0], [heatmap.shape[1], 0], [heatmap.shape[1], heatmap.shape[0]],
                    [0, heatmap.shape[0]]]).astype('float32')

        yc = (y4 + y1 + y3 + y2) / 4
        xc = (x1 + x2 + x3 + x4) / 4
        if orientation == 'horizontal':
            current_link_points = np.array([[
                (xc + (x1 + x2) / 2) / 2, (yc + (y1 + y2) / 2) / 2
            ], [(xc + (x3 + x4) / 2) / 2, (yc + (y3 + y4) / 2) / 2]]) / 2
        else:
            current_link_points = np.array([[
                (xc + (x1 + x4) / 2) / 2, (yc + (y1 + y4) / 2) / 2
            ], [(xc + (x2 + x3) / 2) / 2, (yc + (y2 + y3) / 2) / 2]]) / 2
        character_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]
                                        ]).astype('float32') / 2
        # pylint: disable=unsubscriptable-object
        if previous_link_points is not None:
            if orientation == 'horizontal':
                link_points = np.array([
                    previous_link_points[0], current_link_points[0], current_link_points[1],
                    previous_link_points[1]
                ])
            else:
                link_points = np.array([
                    previous_link_points[0], previous_link_points[1], current_link_points[1],
                    current_link_points[0]
                ])
            ML = cv2.getPerspectiveTransform(
                src=src,
                dst=link_points.astype('float32'),
            )
            linkmap += cv2.warpPerspective(heatmap,
                                            ML,
                                            dsize=(linkmap.shape[1],
                                                    linkmap.shape[0])).astype('float32')
        MA = cv2.getPerspectiveTransform(
            src=src,
            dst=character_points,
        )
        textmap += cv2.warpPerspective(heatmap, MA, dsize=(textmap.shape[1],
                                                            textmap.shape[0])).astype('float32')
        # pylint: enable=unsubscriptable-object
        previous_link_points = current_link_points

    gt = np.concatenate([textmap[..., np.newaxis], linkmap[..., np.newaxis]], axis=2).clip(
        0, 255) / 255
    return gt

def getBoxes(y_pred,
             detection_threshold=0.3,
             text_threshold=0.3,
             link_threshold=0.3,
             size_threshold=10):
    box_groups = []
    for y_pred_cur in y_pred:
        # Prepare data
        textmap = y_pred_cur[..., 0].copy()
        linkmap = y_pred_cur[..., 1].copy()
        img_h, img_w = textmap.shape

        _, text_score = cv2.threshold(textmap,
                                      thresh=text_threshold,
                                      maxval=1,
                                      type=cv2.THRESH_BINARY)
        _, link_score = cv2.threshold(linkmap,
                                      thresh=link_threshold,
                                      maxval=1,
                                      type=cv2.THRESH_BINARY)
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(
            text_score + link_score, 0, 1).astype('uint8'), connectivity=4)
        boxes = []
        for component_id in range(1, n_components):
            # Filter by size
            size = stats[component_id, cv2.CC_STAT_AREA]

            if size < size_threshold:
                continue

            # If the maximum value within this connected component is less than
            # text threshold, we skip it.
            if np.max(textmap[labels == component_id]) < detection_threshold:
                continue

            # Make segmentation map. It is 255 where we find text, 0 otherwise.
            segmap = np.zeros_like(textmap)
            segmap[labels == component_id] = 255
            segmap[np.logical_and(link_score, text_score)] = 0
            x, y, w, h = [
                stats[component_id, key] for key in
                [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
            ]

            # Expand the elements of the segmentation map
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            segmap[sy:ey, sx:ex] = cv2.dilate(
                segmap[sy:ey, sx:ex],
                cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))

            # Make rotated box from contour
            contours = cv2.findContours(segmap.astype('uint8'),
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_SIMPLE)[-2]

            contour = contours[0]
            box = cv2.boxPoints(cv2.minAreaRect(contour))

            # Check to see if we have a diamond
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            else:
                # Make clock-wise order
                box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
            boxes.append(2 * box)
        box_groups.append(np.array(boxes))
    return box_groups
