import cv2
import numpy as np


def find_contours(heatmap, threshold=None, dilation=True, erosion=False):
    """
    Find and sort text line contour based on score link image
    @Parameters:
        - heatmap: score link heatmap image
        - threshold: threshold method, choices=[otsu, adaptive, simple]
        - dilate: whether or not to use dilation
    @Returns:
        - contours: list of contours
        - contour_index: contour sort index
    """
    # Convert to grayscale
    gray = heatmap  # cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    # gray = cv2.GaussianBlur(gray, (5,5), 0)
    height, width = gray.shape[:2]
    # Threshold
    thresh = gray
    if threshold == "otsu":
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif threshold == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    elif threshold == "simple":
        # 180 -> 127
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

    # kernel = np.ones((3, 1), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=1)

    # Dilate
    dilate = thresh
    if dilation:
        # width // 50
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 50, 1))
        dilate = cv2.dilate(thresh, kernel, iterations=3)

    # Erode
    erode = dilate
    if erosion:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        erode = cv2.erode(dilate, kernel, iterations=1)

    # Find and sort contour
    contours = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = [c.squeeze() for c in contours if len(c) > 2]

    contour_left = []
    for c in contours:
        index = np.argsort(c[:, 0])
        contour_left.append(c[index[0], 1])
    contour_index = np.argsort(contour_left)
    return contours, contour_index


def arrange_boxes(boxes, contours, contour_index, ratio, ratio_net=2):
    """
    Arrange word bounding boxes to lines based on contour
    @Parameters:
        - boxes: array of word bounding boxes
        - contours: list of contours
        - contour_index: contour sorting index
        - ratio: image resize ratio
        - ratio_net: CRAFT resize ratio, default=2
    @Returns:
        - lines: line index of each bounding box
    """
    # Calculate distance from each box center to each contour
    centers = (boxes[:, 0:2] + boxes[:, 4:6]) // (2 * ratio * ratio_net)
    distances = np.zeros((len(contours), len(boxes)))

    for idx, c_idx in enumerate(contour_index):
        c = contours[c_idx]
        distances[idx, :] = p_poly_dist(centers, c)
    line_idx = np.argmin(distances, axis=0)

    # Sorting boxes on the same line
    idx = np.argsort(line_idx)

    boxes = boxes[idx]
    _, count = np.unique(line_idx, return_counts=True)

    start = 0
    lines = np.zeros(boxes.shape[0], dtype=np.int)
    for i, c in enumerate(count):
        # Get boxes on the same line
        box_line = boxes[start : start + c]
        # Sorting in order of increasing x
        idx = np.argsort(box_line[:, 0])
        box_line = box_line[idx]
        # Update boxes and move to next line
        boxes[start : start + c] = box_line
        lines[start : start + c] = i
        start += c

    return boxes, lines


def p_poly_dist(p, poly):
    """
    Calculate distance from a list of points to a polygon
    @Parameters:
        - p: array of points [x,y]
        - poly: polygon, array of points [x,y]
    @Returns:
        - d: distance from each point in p to polygon poly

    Algorithm:
        https://www.mathworks.com/matlabcentral/fileexchange/12744-distance-from-points-to-polyline-or-polygon
    """
    # Polygon must have at least 3 points
    assert len(poly) > 2

    # Check if poly is closed, if not then close it
    if ~(poly[0] == poly[-1]).all():
        poly = np.vstack((poly, poly[0]))

    # Get number of point and number of vertices
    Np = len(p)
    Nv = len(poly)

    # Calculate distance from each point to polygon vertices
    dpv = np.hypot(
        np.tile(np.expand_dims(p[:, 0], axis=1), [1, Nv])
        - np.tile(poly[:, 0].T, [Np, 1]),
        np.tile(np.expand_dims(p[:, 1], axis=1), [1, Nv])
        - np.tile(poly[:, 1].T, [Np, 1]),
    )

    # Find closest vertex
    dpv_min = np.amin(dpv, axis=1)
    I_dpv_min = np.argmin(dpv, axis=1)

    # coordinate of consecutive vertices
    P1 = poly[:-1, :]
    P2 = poly[1:, :]
    dv = P2 - P1

    # distance between consecutive vertices
    vds = np.hypot(dv[:, 0], dv[:, 1])

    # Find rotation matrix
    ctheta = dv[:, 0] / vds
    stheta = dv[:, 1] / vds
    Cer = np.zeros((2, 2, Nv - 1))
    Cer[0, 0, :] = ctheta
    Cer[0, 1, :] = stheta
    Cer[1, 0, :] = -stheta
    Cer[1, 1, :] = ctheta

    # rotate P1 vector
    P1r = np.array(
        [P1[:, 0] * ctheta + P1[:, 1] * stheta, -P1[:, 0] * stheta + P1[:, 1] * ctheta]
    ).T

    # compute points new coordinate: rotation -> translation
    Cer21 = Cer[0, :, :]
    Cer22 = Cer[1, :, :]
    Pp1 = np.zeros((2, Np, Nv - 1))
    Pp1[0, :, :] = np.matmul(p, Cer21) - np.tile(
        np.expand_dims(P1r[:, 0].T, axis=0), [Np, 1]
    )
    Pp1[1, :, :] = np.matmul(p, Cer22) - np.tile(
        np.expand_dims(P1r[:, 1].T, axis=0), [Np, 1]
    )

    # Check if projection fall inside segment
    r = Pp1[0, :, :]
    cr = Pp1[1, :, :]
    is_in_seg = (r > 0) & (r < np.tile(np.expand_dims(vds, axis=0), [Np, 1]))

    # find minimum distance
    B = np.full((Np, Nv - 1), np.Inf)
    B[is_in_seg] = cr[is_in_seg]
    B = np.abs(B)

    cr_min = np.amin(B, axis=1)
    I_cr_min = np.argmin(B, axis=1)

    cond1 = cr_min == np.Inf
    cond2 = (I_cr_min != I_dpv_min) & ((cr_min - dpv_min) > 0)
    is_vertex = cond1 | cond2

    dmin = cr_min
    dmin[is_vertex] = dpv_min[is_vertex]

    return dmin
