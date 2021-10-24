import copy
import imageio

import cv2
import numpy as np
import torch
import dgl

import configs as cf
from models.kie.gated_gcn import GatedGCNNet
from backend.backend_utils import timer


def load_gate_gcn_net(device, checkpoint_path):
    net_params = {}
    net_params["in_dim_text"] = len(cf.alphabet)
    net_params["in_dim_node"] = 10
    net_params["in_dim_edge"] = 2
    net_params["hidden_dim"] = 512
    net_params["out_dim"] = 384
    net_params["n_classes"] = 5
    net_params["in_feat_dropout"] = 0.1
    net_params["dropout"] = 0.0
    net_params["L"] = 4
    net_params["readout"] = True
    net_params["graph_norm"] = True
    net_params["batch_norm"] = True
    net_params["residual"] = True
    net_params["device"] = "cuda"
    net_params["OHEM"] = 3

    model = GatedGCNNet(net_params)

    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(cf.device)
    )
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def make_text_encode(text):
    text_encode = []
    for t in text.upper():
        if t not in cf.alphabet:
            text_encode.append(cf.alphabet.index(" "))
        else:
            text_encode.append(cf.alphabet.index(t))
    return np.array(text_encode)


def prepare_data(cells, text_key="vietocr_text"):
    texts = []
    text_lengths = []
    polys = []
    for cell in cells:
        text = cell[text_key]
        text_encode = make_text_encode(text)
        text_lengths.append(text_encode.shape[0])
        texts.append(text_encode)

        poly = copy.deepcopy(cell["poly"].tolist())
        poly.append(np.max(poly[0::2]) - np.min(poly[0::2]))
        poly.append(np.max(poly[1::2]) - np.min(poly[1::2]))
        poly = list(map(int, poly))
        polys.append(poly)

    texts = np.array(texts, dtype=object)
    text_lengths = np.array(text_lengths)
    polys = np.array(polys)
    return texts, text_lengths, polys


def prepare_pipeline(boxes, edge_data, text, text_length):
    box_min = boxes.min(0)
    box_max = boxes.max(0)

    boxes = (boxes - box_min) / (box_max - box_min)
    boxes = (boxes - 0.5) / 0.5

    edge_min = edge_data.min(0)
    edge_max = edge_data.max(0)

    edge_data = (edge_data - edge_min) / (edge_max - edge_min)
    edge_data = (edge_data - 0.5) / 0.5

    return boxes, edge_data, text, text_length


@timer
def prepare_graph(cells):
    texts, text_lengths, boxes = prepare_data(cells)

    origin_boxes = boxes.copy()
    node_nums = text_lengths.shape[0]

    src = []
    dst = []
    edge_data = []
    for i in range(node_nums):
        for j in range(node_nums):
            if i == j:
                continue

            edata = []
            # y distance
            y_distance = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
            # w = boxes[i, 8]
            h = boxes[i, 9]
            if np.abs(y_distance) > 3 * h:
                continue

            x_distance = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
            edata.append(y_distance)
            edata.append(x_distance)

            edge_data.append(edata)
            src.append(i)
            dst.append(j)

    edge_data = np.array(edge_data)
    g = dgl.DGLGraph()
    g.add_nodes(node_nums)
    g.add_edges(src, dst)

    boxes, edge_data, text, text_length = prepare_pipeline(
        boxes, edge_data, texts, text_lengths
    )
    boxes = torch.from_numpy(boxes).float()
    edge_data = torch.from_numpy(edge_data).float()

    tab_sizes_n = g.number_of_nodes()
    tab_snorm_n = torch.FloatTensor(tab_sizes_n, 1).fill_(1.0 / float(tab_sizes_n))
    snorm_n = tab_snorm_n.sqrt()

    tab_sizes_e = g.number_of_edges()
    tab_snorm_e = torch.FloatTensor(tab_sizes_e, 1).fill_(1.0 / float(tab_sizes_e))
    snorm_e = tab_snorm_e.sqrt()

    max_length = text_lengths.max()
    new_text = [
        np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), "constant"), axis=0)
        for t in text
    ]
    texts = np.concatenate(new_text)

    texts = torch.from_numpy(np.array(texts))
    text_length = torch.from_numpy(np.array(text_length))

    graph_node_size = [g.number_of_nodes()]
    graph_edge_size = [g.number_of_edges()]

    return (
        g,
        boxes,
        edge_data,
        snorm_n,
        snorm_e,
        texts,
        text_length,
        origin_boxes,
        graph_node_size,
        graph_edge_size,
    )


@timer
def run_predict(gcn_net, merged_cells, device="cpu"):

    (
        batch_graphs,
        batch_x,
        batch_e,
        batch_snorm_n,
        batch_snorm_e,
        text,
        text_length,
        boxes,
        graph_node_size,
        graph_edge_size,
    ) = prepare_graph(merged_cells)

    batch_graphs = batch_graphs.to(device)
    batch_x = batch_x.to(device)
    batch_e = batch_e.to(device)

    text = text.to(device)
    text_length = text_length.to(device)
    batch_snorm_e = batch_snorm_e.to(device)
    batch_snorm_n = batch_snorm_n.to(device)

    batch_graphs = batch_graphs.to(device)
    batch_scores = gcn_net.forward(
        batch_graphs,
        batch_x,
        batch_e,
        text,
        text_length,
        batch_snorm_n,
        batch_snorm_e,
        graph_node_size,
        graph_edge_size,
    )
    return batch_scores, boxes


@timer
def postprocess_scores(batch_scores, score_ths=0.98, get_max=False):
    values, preds = [], []
    batch_scores = batch_scores.cpu().softmax(1)
    for score in batch_scores:
        _score = score.detach().cpu().numpy()
        values.append(_score.max())
        pred_index = np.argmax(_score)
        if get_max:
            preds.append(pred_index)
        else:
            if pred_index != 0 and _score.max() >= score_ths:
                preds.append(pred_index)
            else:
                preds.append(0)

    preds = np.array(preds)
    return values, preds


@timer
def postprocess_write_info(merged_cells, preds, text_key="vietocr_text"):
    # 1/2/3/4
    # 'ADDRESS', 'SELLER', 'TIMESTAMP', 'TOTAL_COST'
    kie_info = dict()
    preds = np.array(preds)
    for i in range(1, 5):
        indexes = np.where(preds == i)[0]
        if len(indexes) > 0:
            text_output = " ".join(merged_cells[index][text_key] for index in indexes)
            kie_info[cf.node_labels[i].title()] = text_output
    return kie_info


@timer
def vis_kie_pred(img, preds, values, boxes, save_path):
    vis_img = img.copy()
    length = preds.shape[0]
    for i in range(length):

        pred_id = preds[i]
        if pred_id != 0:
            msg = "{}-{}".format(cf.node_labels[preds[i]], round(float(values[i]), 2))
            color = (0, 0, 255)

            info = boxes[i]
            box = np.array(
                [
                    [int(info[0]), int(info[1])],
                    [int(info[2]), int(info[3])],
                    [int(info[4]), int(info[5])],
                    [int(info[6]), int(info[7])],
                ]
            )
            cv2.polylines(vis_img, [box], 1, (255, 0, 0))
            cv2.putText(
                vis_img,
                msg,
                (int(info[0]), int(info[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    imageio.imwrite(save_path, vis_img)
    return vis_img
