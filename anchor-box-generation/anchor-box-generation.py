import numpy as np
import math
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    anchors = []
    stride = image_size / feature_size
    for i in range(feature_size):
        for j in range(feature_size):
            for scale in scales:
                for ratio in aspect_ratios:
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    anchor = list([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                    anchors.append(anchor)
    return anchors
        