# Key Information Extraction from Scanned Invoices

> Key information extraction from invoice document with Graph Convolution Network

- Related blog post from my Viblo account: https://viblo.asia/p/djeZ1yPGZWz 

### Models

- Background subtraction: U2Net
- Image alignment: based-on output of text-detection & cv2
- Text detection: CRAFT and an in-house text-detection model
- Text recognition: VietOCR and an in-house text-recognition model
- KIE: Graph Convolution

### Pretrained model

- [Google Drive](https://drive.google.com/file/d/10qveX-0PpS9TtJdFX0ID0ZqNLw3X3jUt/view?usp=sharing)

### Data

- MC-OCR, a Vietnamese receipts dataset: https://aihub.vn/competitions/1
- Preprocessed data: [Google Drive](https://drive.google.com/file/d/14zXVyu-wA_Rg0bb4Vmz3ElFkATI25kbH/view?usp=sharing)

### Pipeline

TODO

### Command

- Create virtual environment using conda or virtualenv

```bash
# with virtualenv
virtualenv -p python3 invoice_env
# activate environment
source invoice_env/bin/activate
# install prerequisite libraries
pip install -r requirements.txt
```

```bash
# 1st command, run API
make serve
# 2nd command, run web-gui with streamlit
make runapp
```

Then access the localhost server at: 0.0.0.0:7778

### Preview

![](./assets/demo_kie_invoice.png)

### TODO

- Add preprocess data script

### Reference

- MC-OCR dataset: https://aihub.vn/competitions/1
- U2Net: https://github.com/xuebinqin/U-2-Net
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- VietOCR: https://github.com/pbcquoc/vietocr
- Benchmarking GNNs: https://github.com/graphdeeplearning/benchmarking-gnns
