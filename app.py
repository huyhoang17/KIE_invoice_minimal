import io
import os
import time
import uuid
import json
import base64
import imageio

import torch
import numpy as np
from PIL import Image
import streamlit as st
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import configs as cf
from backend.kie.kie_utils import (
    load_gate_gcn_net,
    run_predict,
    vis_kie_pred,
    postprocess_scores,
    postprocess_write_info,
)
from backend.backend_utils import create_merge_cells, get_request_api

st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    gcn_net = load_gate_gcn_net(cf.device, cf.kie_weight_path)
    config = Cfg.load_config_from_name("vgg_seq2seq")
    config["cnn"]["pretrained"] = False
    config["device"] = cf.device
    config["predictor"]["beamsearch"] = False
    detector = Predictor(config)

    return gcn_net, detector


gcn_net, detector = load_model()


def infer(img_fp, save_dir):

    json_res = json.loads(get_request_api(img_fp))
    api_runtime = json_res["api_runtime"]
    api_random_id = json_res["random_id"]

    start = time.time()
    imgdata = base64.b64decode(json_res["image"])
    pil_img = Image.open(io.BytesIO(imgdata))
    img = np.array(pil_img)

    cells = json_res["cells"]
    group_ids = np.array([i["group_id"] for i in json_res["cells"]])
    # merge adjacent text-boxes
    merged_cells = create_merge_cells(
        detector, img, cells, group_ids, merge_text=cf.merge_text
    )
    batch_scores, boxes = run_predict(gcn_net, merged_cells, device=cf.device)

    # 2 options: get max score or filter categories by threshold
    values, preds = postprocess_scores(
        batch_scores, score_ths=cf.score_ths, get_max=cf.get_max
    )
    kie_info = postprocess_write_info(merged_cells, preds)

    delta_time = time.time() - start
    print("[TIME]runtime: {}".format(delta_time))

    # visualize prediction
    save_path = os.path.join(save_dir, "{}.jpg".format(api_random_id))
    vis_img = vis_kie_pred(img, preds, values, boxes, save_path)

    return vis_img, kie_info, delta_time + api_runtime


def show_gpu_info():
    allocated_mem = "[GPU]Max Memory Allocated {} MB".format(
        torch.cuda.max_memory_allocated(device="cuda") / 1024.0 / 1024.0
    )
    cached_mem = "[GPU]Max Memory Cached {} MB".format(
        torch.cuda.max_memory_reserved(device="cuda") / 1024.0 / 1024.0
    )
    text_write = "{}\n{}".format(allocated_mem, cached_mem)
    st.text(text_write)


def main():

    st.title("Invoice extraction")

    option_col1, option_col2 = st.columns(2)
    col1, col2, col3 = st.columns(3)
    random_id = str(uuid.uuid4())

    with option_col1:
        with st.form("form1", clear_on_submit=True):
            content_file = st.file_uploader(
                "Upload your image here", type=["jpg", "jpeg", "png"]
            )
            submit = st.form_submit_button("Upload")
            if content_file is not None:
                pil_img = Image.open(content_file)
                img = np.array(pil_img)
                raw_img_fp = os.path.join(
                    cf.raw_img_dir, "{}_raw.png".format(random_id)
                )

                if submit:
                    print(">" * 100)
                    wait_text = st.text("Please wait...")
                    imageio.imwrite(raw_img_fp, img)
                    vis_img, kie_info, total_runtime = infer(
                        raw_img_fp, cf.result_img_dir
                    )
                    wait_text.empty()

                    with col1:
                        st.image(img)
                    with col2:
                        st.image(vis_img)
                    with col3:
                        text_write = "".join(
                            ["{}: {}\n".format(k, v) for k, v in kie_info.items()]
                        )
                        st.text(text_write)
                        st.markdown("---")
                        show_gpu_info()
                        st.markdown("---")
                        total_runtime = round(float(total_runtime), 4)
                        st.text("Total runtime: {}s".format(total_runtime))

    with option_col2:
        with st.form("form2", clear_on_submit=True):
            test_img_fns = os.listdir(cf.img_dir)
            test_img_fns.insert(0, None)
            test_img_fn = st.selectbox("Or select test image here", test_img_fns)
            submit = st.form_submit_button("Upload")

            if test_img_fn is not None:
                with option_col2:
                    select_text = st.text("You selected: {}".format(test_img_fn))

                test_img_fp = os.path.join(cf.img_dir, test_img_fn)
                img = np.array(Image.open(test_img_fp))

                if submit:
                    print(">" * 100)
                    wait_text = st.text("Please wait...")
                    vis_img, kie_info, total_runtime = infer(
                        test_img_fp, cf.result_img_dir
                    )
                    wait_text.empty()

                    with col1:
                        st.image(img)
                    with col2:
                        st.image(vis_img)
                    with col3:
                        text_write = "".join(
                            ["{}: {}\n".format(k, v) for k, v in kie_info.items()]
                        )
                        st.text(text_write)
                        st.markdown("---")
                        show_gpu_info()
                        st.markdown("---")
                        total_runtime = round(float(total_runtime), 4)
                        st.text("Total runtime: {}s".format(total_runtime))

                if select_text is not None:
                    select_text.empty()


if __name__ == "__main__":
    main()
