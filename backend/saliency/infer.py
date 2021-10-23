import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable

import configs as cf
from backend.backend_utils import timer
from backend.saliency.data_loader import SalObjDataset, RescaleT, ToTensorLab, normPRED


@timer
def run_saliency(net, img):
    test_salobj_dataset = SalObjDataset(
        img_name_list=[img],
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
    )

    data_test = test_salobj_dataset[0]
    inputs_test = data_test["image"]
    if len(inputs_test.shape) == 3:
        inputs_test = inputs_test.unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)
    inputs_test = Variable(inputs_test)

    d1, *_ = net(inputs_test.to(cf.device))
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)
    np_img = pred.detach().cpu().numpy().squeeze()

    h_origin, w_origin, _ = img.shape
    mask_img = cv2.resize(np_img, (w_origin, h_origin))
    ths = cf.saliency_ths
    mask_img[mask_img > ths] = 1.0
    mask_img[mask_img <= ths] = 0.0
    mask_img = (mask_img * 255).astype(np.uint8)

    return mask_img
