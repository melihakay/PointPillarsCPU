# This file is modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/iou3d/iou3d_utils.py

import torch
from .iou3d_op import boxes_overlap_bev_gpu, boxes_iou_bev_gpu, nms_gpu, nms_normal_gpu


def boxes_overlap_bev(boxes_a, boxes_b):
    """Calculate boxes Overlap in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_overlap (torch.Tensor): Overlap result with shape (M, N).
    """
    ans_overlap = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_overlap)

    return ans_overlap


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou

def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """
    Modified NMS implementation to handle both CPU and GPU cases.
    Maintains compatibility with original API.
    """
    # Sort based on scores
    if pre_maxsize is not None:
        order = scores.sort(0, descending=True)[1]
        order = order[:pre_maxsize]
    else:
        order = torch.sort(scores, descending=True)[1]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long, device=boxes.device)
    
    if boxes.is_cuda:
        # For CUDA tensors, use the original nms_gpu implementation
        device_idx = boxes.device.index if boxes.device.index is not None else 0
        num_out = nms_gpu(boxes, keep, thresh, device_idx)
    else:
        # For CPU tensors, use torchvision's nms implementation
        import torchvision
        keep_cpu = torchvision.ops.nms(
            boxes[:, :4],  # Use only x1,y1,x2,y2 for NMS
            scores,
            thresh
        )
        num_out = len(keep_cpu)
        keep[:num_out] = keep_cpu

    # Handle the output consistently with the original implementation
    keep = order[keep[:num_out]].contiguous()
    
    if post_max_size is not None:
        keep = keep[:post_max_size]
    
    return keep


def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = nms_normal_gpu(boxes, keep, thresh,
                                        boxes.device.index)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()
