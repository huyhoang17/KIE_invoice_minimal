import os
import time
import json
import uuid
import base64
from io import BytesIO

import imageio
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from backend.models import load_text_detect, load_text_recognize, load_saliency
from backend.backend_utils import (
    NpEncoder,
    run_ocr,
    make_warp_img,
    resize_and_pad,
    get_group_text_line,
)
from backend.text_detect.config import craft_config
from backend.saliency.infer import run_saliency
import configs as cf

app = FastAPI()
net = load_saliency()
detector = load_text_recognize()
text_detector = load_text_detect()


def infer(img, random_id):

    img = resize_and_pad(img, size=1024, pad=False)
    imageio.imwrite(os.path.join(cf.raw_img_dir, "{}.jpg".format(random_id)), img)

    # SALIENCY PREDICTION
    mask_img = run_saliency(net, img)
    img[~mask_img.astype(bool)] = 0.0

    # TRANSFORM AND WRAP IMAGE
    warped_img = make_warp_img(img, mask_img)
    sub_img_fp = os.path.join(cf.cropped_img_dir, "{}.jpg".format(random_id))
    imageio.imwrite(sub_img_fp, warped_img)

    # OCR
    cells, heatmap, textboxes = run_ocr(
        text_detector, detector, warped_img, craft_config
    )
    _, lines = get_group_text_line(heatmap, textboxes)
    for line_id, cell in zip(lines, cells):
        cell["group_id"] = line_id

    # CROP IMG AND UPDATE FINAL CELLS INFO
    sub_h_img, sub_w_img, _ = warped_img.shape
    img_info = dict()
    img_info["h_origin"] = sub_h_img
    img_info["w_origin"] = sub_w_img
    img_info["cells"] = cells

    with open(sub_img_fp, "rb") as image_file:
        encoded_image_string = base64.b64encode(image_file.read()).decode("utf-8")
    img_info["image"] = encoded_image_string
    return img_info


@app.post("/mcocr/")
async def mcocr(image: UploadFile = File(...)):

    print(">" * 100)
    total_start_time = time.time()
    random_id = str(uuid.uuid4())
    image = np.array(Image.open(BytesIO(await image.read())))

    start = time.time()
    img_info = infer(image, random_id)
    delta_time = time.time() - start
    print("[TIME]runtime: {}".format(delta_time))

    img_info["api_runtime"] = delta_time
    img_info["random_id"] = random_id
    response = json.dumps(img_info, cls=NpEncoder, ensure_ascii=False)
    print("[TIME]total time API: {}".format(time.time() - total_start_time))

    return response
