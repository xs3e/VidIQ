import torch
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2)

from VidIQ.dnn_model.util import (f1_score, scale_boxes_of_results,
                                   show_detections)

faster_rcnn_running = None
faster_rcnn_weights = None


def load_faster_rcnn(index=1, rank=0):
    score_thresh = 0.5
    if index == 1:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(
            weights=weights, box_score_thresh=score_thresh)
    elif index == 2:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=score_thresh)
    model = model.to(rank)
    model.eval()
    return weights, model


def run_faster_rcnn(imgs, inputs, gt_imgs, gt_inputs, show=True, online=False):
    global faster_rcnn_running, faster_rcnn_weights
    if faster_rcnn_running is None:
        faster_rcnn_weights, faster_rcnn_running = load_faster_rcnn()

    with torch.no_grad():
        results = faster_rcnn_running(inputs)
        if online is False:
            gt_results = faster_rcnn_running(gt_inputs)

    if show:
        show_detections(faster_rcnn_weights, imgs, results, 0)
        show_detections(faster_rcnn_weights, gt_imgs, gt_results, 0)

    if online is False:
        # print(results[0])
        results = scale_boxes_of_results(results, inputs, gt_inputs)
        acc = f1_score(results, gt_results)
        print(acc)
        return acc
    else:
        return results
