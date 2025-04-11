import torch
from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                          keypointrcnn_resnet50_fpn)

from VidIQ.dnn_model.util import (distance_accuracy, scale_keypoints,
                                   show_keypoints)

keypoint_rcnn_running = None
keypoint_rcnn_weights = None


def load_keypoint_rcnn(rank=0):
    score_thresh = 0.5
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(
        weights=weights, box_score_thresh=score_thresh)
    model = model.to(rank)
    model.eval()
    return weights, model


def run_keypoint_rcnn(imgs, inputs, gt_imgs, gt_inputs, show=True, online=False):
    global keypoint_rcnn_running, keypoint_rcnn_weights
    if keypoint_rcnn_running is None:
        keypoint_rcnn_weights, keypoint_rcnn_running = load_keypoint_rcnn()

    with torch.no_grad():
        results = keypoint_rcnn_running(inputs)
        if online is False:
            gt_results = keypoint_rcnn_running(gt_inputs)

    if show:
        show_keypoints(imgs, results, 0)
        show_keypoints(gt_imgs, gt_results, 0)

    if online is False:
        # print(results[0])
        results = scale_keypoints(results, inputs, gt_inputs)
        acc = distance_accuracy(results, gt_results)
        print(acc)
        return acc
    else:
        return results
