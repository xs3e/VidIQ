import torch
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                          MaskRCNN_ResNet50_FPN_Weights,
                                          maskrcnn_resnet50_fpn,
                                          maskrcnn_resnet50_fpn_v2)

from VidIQ.dnn_model.util import (miou_accuracy, scale_segmentation_masks,
                                   show_segmentations)

mask_rcnn_running = None
mask_rcnn_weights = None


def load_mask_rcnn(index=1, rank=0):
    score_thresh = 0.5
    if index == 1:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(
            weights=weights, box_score_thresh=score_thresh)
    elif index == 2:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=score_thresh)
    model = model.to(rank)
    model.eval()
    return weights, model


def run_mask_rcnn(imgs, inputs, gt_imgs, gt_inputs, show=True, online=False):
    global mask_rcnn_running, mask_rcnn_weights
    if mask_rcnn_running is None:
        mask_rcnn_weights, mask_rcnn_running = load_mask_rcnn()

    with torch.no_grad():
        results = mask_rcnn_running(inputs)
        if online is False:
            gt_results = mask_rcnn_running(gt_inputs)

    if show:
        show_segmentations(imgs, results, 0)
        show_segmentations(gt_imgs, gt_results, 0)

    if online is False:
        # print(results[0])
        results = scale_segmentation_masks(results, inputs, gt_inputs)
        acc = miou_accuracy(results, gt_results)
        print(acc)
        return acc
    else:
        return results
