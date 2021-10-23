import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import configs as cf
from models.saliency.u2net import U2NETP
from backend.text_detect.craft_utils import get_detector


def load_text_detect():
    text_detector = get_detector(cf.text_detection_weights_path, cf.device)
    return text_detector


def load_saliency():
    net = U2NETP(3, 1)
    net = net.to(cf.device)
    net.load_state_dict(torch.load(cf.saliency_weight_path, map_location=cf.device))
    net.eval()
    return net


def load_text_recognize():
    config = Cfg.load_config_from_name("vgg_seq2seq")
    config["cnn"]["pretrained"] = False
    config["device"] = cf.device
    config["predictor"]["beamsearch"] = False
    detector = Predictor(config)

    return detector
