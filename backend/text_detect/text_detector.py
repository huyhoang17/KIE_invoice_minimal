from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from models.text_detect.craft import CRAFT
from backend.text_detect.craft_utils import (
    adjustResultCoordinates,
    getDetBoxes,
)
from backend.text_detect.imgproc import (
    normalizeMeanVariance,
    resize_aspect_ratio,
)


def scale(x, s):
    """Scales x by scaling factor s.
    Parameters
    ----------
    x : float
    s : float
    Returns
    -------
    x : float
    """
    x *= s
    return x


def rescale_coordinate(k, factors):
    """Translates and scales resized image coordinate  to original image
    coordinate space.
    Parameters
    ----------
    k : tuple
        Tuple (x1, y1, x2, y2) representing table bounding box where
        (x1, y1) -> lt and (x2, y2) -> rb in PDFMiner coordinate
        space.
    factors : tuple
        Tuple (scaling_factor_x, scaling_factor_y, pdf_y) where the
        first two elements are scaling factors and pdf_y is height of
        pdf.
    Returns
    -------
    knew : tuple
        Tuple (x1, y1, x2, y2) representing table bounding box where
        (x1, y1) -> lt and (x2, y2) -> rb in OpenCV coordinate
        space.
    """
    x1, y1, x2, y2 = k
    scaling_factor_x, scaling_factor_y, pdf_y = factors
    x1 = scale(x1, scaling_factor_x)
    y1 = scale(y1, scaling_factor_y)
    x2 = scale(x2, scaling_factor_x)
    y2 = scale(y2, scaling_factor_y)
    knew = (int(x1), int(y1), int(x2), int(y2))

    return knew


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(
    canvas_size,
    mag_ratio,
    net,
    image,
    text_threshold,
    link_threshold,
    low_text,
    poly,
    device,
    estimate_num_chars=False,
):
    # resize
    img_resized, target_ratio, size_heatmap, delta_h, delta_w = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys, mapper = getDetBoxes(
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
        poly,
        estimate_num_chars,
    )

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    if estimate_num_chars:
        boxes = list(boxes)
        polys = list(polys)
    for k in range(len(polys)):
        if estimate_num_chars:
            boxes[k] = (boxes[k], mapper[k])
        if polys[k] is None:
            polys[k] = boxes[k]

    return y, boxes, polys, delta_h, delta_w


def get_detector(trained_model, device="cpu"):
    net = CRAFT()

    if device == "cpu":
        net.load_state_dict(
            copyStateDict(torch.load(trained_model, map_location=device))
        )
    else:
        net.load_state_dict(
            copyStateDict(torch.load(trained_model, map_location=device))
        )
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = False

    net.eval()
    return net


def get_textbox(
    detector,
    image,
    canvas_size,
    mag_ratio,
    text_threshold,
    link_threshold,
    low_text,
    poly,
    device,
    optimal_num_chars=None,
):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    y, bboxes, polys, delta_h, delta_w = test_net(
        canvas_size,
        mag_ratio,
        detector,
        image,
        text_threshold,
        link_threshold,
        low_text,
        poly,
        device,
        estimate_num_chars,
    )

    # bboxes, polys (no_box, 4, 2), dtype=float32
    if estimate_num_chars:
        polys = [
            p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))
        ]

    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    return y, result
