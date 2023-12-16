import torch
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config import DEVICE, S


def iou(pred, tar):
    # here pred is assumed to be the (x,y,width,height) of the bounding box
    # and tar is assumed to have the same format
    # left top corner coordinates
    x1_pred, y1_pred = pred[..., 0] - pred[..., 2], pred[..., 1] - pred[..., 3]
    x1_tar, y1_tar = tar[..., 0] - tar[..., 2], tar[..., 1] - pred[..., 3]
    # right bottom corner coordinates
    x2_pred, y2_pred = pred[..., 0] + pred[..., 2], pred[..., 1] + pred[..., 3]
    x2_tar, y2_tar = tar[..., 0] + tar[..., 2], tar[..., 1] + tar[..., 3]

    # calculate corner coordinates of the intersection rectangle
    x1 = torch.max(x1_pred, x1_tar)
    y1 = torch.max(y1_pred, y1_tar)
    x2 = torch.min(x2_pred, x2_tar)
    y2 = torch.min(y2_pred, y2_tar)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    tar_area = (x2_tar - x1_tar) * (y2_tar - y1_tar)
    union = pred_area + tar_area - intersection
    return intersection / union


def non_max_suppression(bboxes, iou_thres, prob_thres):
    # Input: bboxes in a single img
    # Output: filtered bboxes in this img
    # might have multiple bboxes for a single object, use NMS to clean up

    # expected bboxes format: [[cls_id,prob,x,y,w,h], [], ...]
    # first pass: if bboxes having higher probability indicating they contain objects
    bboxes = bboxes.view(-1, 6)
    bboxes = [box for box in bboxes if box[1].item() > prob_thres]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    nmsed_bboxes = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        nmsed_bboxes.append(chosen_box)
        # second pass: if pass if not the same type as the current box(having the highest probability)
        # or pass if it doesn't overlap too much with the current box
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or
               iou(chosen_box[2:], box[2:]) < iou_thres
        ]
    return nmsed_bboxes


def mean_average_precision(pred, tar, iou_thres):
    # expected pred format: [[img_id,cls_id,prob,x,y,w,h],[],...] in img-relative coordinates
    ap = []  # Average Precision
    # calculate AP for each class out of a total of 20 and return the average
    for cls in range(1, 21):  # 1-indexed
        # Step 1: get all bboxes in this class
        # get prediction&target bboxes of the current class
        pred_box_same_cls = []
        tar_box_same_cls = []
        is_tar_empty = 0
        for box in pred:
            if box[1] == cls:
                pred_box_same_cls.append(box)
        for box in tar:
            if box[1] == cls:
                tar_box_same_cls.append(box)
                is_tar_empty = 1
        if is_tar_empty == 0:
            continue
        # Counter creates a dictionary with key:img_idx, value: # of bboxes in that img
        bboxes_record = Counter([int(box[0].tolist()) for box in tar_box_same_cls])
        # turn value into a zero tensor with the length of the original value
        # ex: (0:2)->(0:torch.sensor([0,0]))
        # this is to keep track of which target bboxes have been match to prediction bboxes
        # 0 if not matched, 1 if matched
        for k, v in bboxes_record.items():
            bboxes_record[k] = torch.zeros(v)
        # Step 2: Sort all prediction bboxes in descent order of their probability scores
        # so that we deal with possible bboxes first
        pred_box_same_cls.sort(key=lambda x: x[2], reverse=True)

        # Step 3: For certain prediction bbox, check target bboxes in the same img as the prediction bbox
        # find the max_iou, and then check
        # if iou > iou_thres
        tp = torch.zeros(len(pred_box_same_cls))
        fp = torch.zeros(len(pred_box_same_cls))
        for pred_idx, pred_bbox in enumerate(pred_box_same_cls):
            corres_tar_bboxes = [bbox for bbox in tar_box_same_cls if bbox[0] == pred_bbox[0]]
            # find max iou(best match of this pred_bbbox)
            max_iou = -1
            max_iou_idx = -1
            for tar_idx, tar_bbox in enumerate(corres_tar_bboxes):
                current_iou = iou(pred_bbox[..., 3:], tar_bbox[..., 3:])
                if current_iou > max_iou:
                    max_iou = current_iou
                    max_iou_idx = tar_idx
            # check if best_iou exceeds iou_thres
            if max_iou > iou_thres:
                # consider a match only if it's not been matched before
                if bboxes_record[int(pred_bbox[0])][max_iou_idx] == 0:
                    tp[pred_idx] = 1
                    bboxes_record[int(pred_bbox[0])][max_iou_idx] == 1
                # else it's valid iou number but recorded before, so it's a false positive
                else:
                    fp[pred_idx] = 1
            # max_iou < iou_thres, no valid overlapping, it's a false positive
            else:
                fp[pred_idx] = 1

        # Step 4: using the TP/FP data for all imgs for the current class and current iou_thres
        # calculate the AP
        # tp is a vector, ex: [1, 1, 0, 1, 0,...] indicating if the bbox is tp or not
        # tp_cumsum is still a vector calculating the cumulative sum, ex: [1, 2, 2, 3, 3]
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        # recall = tp/(tp+fn), precision = tp/(tp+fp)
        recall = tp_cumsum / (len(tar_box_same_cls) + 1e-6)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        # concatenate recall with 0 and precision with 1 to add a [0,1] point in recall-precision graph
        # because [0,1] is the point where we should start when we later perform numeric integration
        recall = torch.cat((torch.tensor([0]), recall))
        precision = torch.cat((torch.tensor([1]), precision))
        ap.append(torch.trapz(precision, recall))
    # by now, we have AP for 20 classes and for 1 iou_thres
    # for metric like iou@0.5:0.05:0.95, call current function multiple times
    return sum(ap) / len(ap)


def bbox_cell2img(pred_bboxes, tar_bboxes):
    # Input: prediction bboxes and target bboxes can have general combination of (N,S,S,30/25)
    # Output: rescaled prediction bboxes and target bboxes have (N, S*S, 6) shape
    # rescale bboxes coordinates from cell-relative(which is the output from the model) to img-relative
    pred_bboxes = pred_bboxes.view(pred_bboxes.shape[0], S, S, 30)
    pred_bbox1 = pred_bboxes[..., 21:25]
    pred_bbox2 = pred_bboxes[..., 26:30]
    # the output from the model will contain all bounding boxes, but only the better one is used for mAP
    # therefore, we only need to consider the better one
    scores = torch.cat((pred_bboxes[..., 20].unsqueeze(0), pred_bboxes[..., 25].unsqueeze(0)))
    better_bbox_idx = scores.argmax(0).unsqueeze(-1)
    better_bbox = pred_bbox1 * (1 - better_bbox_idx) + pred_bbox2 * better_bbox_idx
    cell_idx = torch.arange(S).repeat(pred_bboxes.shape[0], S, 1).unsqueeze(-1).to(DEVICE)
    pred_bboxes = torch.cat(
        ((pred_bboxes[..., :20].argmax(-1) + 1).unsqueeze(-1),
         torch.max(pred_bboxes[..., 20], pred_bboxes[..., 25]).unsqueeze(-1),
         (better_bbox[..., 0].unsqueeze(-1) + cell_idx) / S,
         (better_bbox[..., 1].unsqueeze(-1) + cell_idx).permute(0, 2, 1, 3) / S,
         better_bbox[..., 2:4] / S), dim=-1
    )
    if tar_bboxes.tolist() == []:
        return pred_bboxes, []
    tar_bboxes = tar_bboxes.view(tar_bboxes.shape[0], S * S, -1)
    cls_id = torch.zeros((tar_bboxes.shape[0], S * S)).to(DEVICE)
    for i in range(tar_bboxes.shape[0]):
        for j in range(S * S):
            if tar_bboxes[i, j, 20] == 0:
                cls_id[i, j] = 0
            else:
                cls_id[i, j] = tar_bboxes[i, j, :20].argmax(-1) + 1

    tar_bboxes = tar_bboxes.view(tar_bboxes.shape[0], S, S, -1)
    tar_x, tar_y = torch.zeros((tar_bboxes.shape[0], S, S, 1)).to(DEVICE), \
                   torch.zeros((tar_bboxes.shape[0], S, S, 1)).to(DEVICE)
    for n in range(tar_bboxes.shape[0]):
        for i in range(S):
            for j in range(S):
                tar_x[n, i, j, 0] = 0 if tar_bboxes[n, i, j, 20].item() == 0 \
                    else (j + 1 + tar_bboxes[n, i, j, 21]) / S
                tar_y[n, i, j, 0] = 0 if tar_bboxes[n, i, j, 20].item() == 0 \
                    else (i - 1 + tar_bboxes[n, i, j, 22]) / S

    tar_bboxes = torch.cat(
        (cls_id.view(cls_id.shape[0], S, S).unsqueeze(-1),
         tar_bboxes[..., 20].unsqueeze(-1),
         tar_x,
         tar_y,
         tar_bboxes[..., 23:] / S), dim=-1
    )
    pred_bboxes = pred_bboxes.view(pred_bboxes.shape[0], -1, 6)
    tar_bboxes = tar_bboxes.view(tar_bboxes.shape[0], -1, 6)
    return pred_bboxes, tar_bboxes


def get_bboxes(dataloader, model, iou_thres, prob_thres):
    # this function returns pred_bboxes and tar_bboxes for mAP calculation uses.
    pred_bboxes = []
    tar_bboxes = []
    train_idx = 0
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # scale cell-relative coordinates to img-relative so that we can use in mAP calculation
            batch_pred_bboxes, batch_tar_bboxes = bbox_cell2img(model(x), y)
            # in every img in the current batch, use non-max-suppression to clean the predictions
            for i in range(x.shape[0]):
                nms_batch_pred_bboxes = non_max_suppression(batch_pred_bboxes[i], iou_thres, prob_thres)
                # append every left bboxes to the result set
                for box in nms_batch_pred_bboxes:
                    pred_bboxes.append(torch.cat((torch.tensor(train_idx).unsqueeze(-1).to(DEVICE), box), dim=-1))
                for box in batch_tar_bboxes[i]:
                    if box[1] > prob_thres:
                        tar_bboxes.append(torch.cat((torch.tensor(train_idx).unsqueeze(-1).to(DEVICE), box), dim=-1))
                train_idx += 1
        model.train()
    return pred_bboxes, tar_bboxes


def save(state, filename="100ex.tar"):
    print("saving---------------")
    torch.save(state, filename)


def load(saved_model, model, optimizer):
    print("loading--------------")
    model.load_state_dict(saved_model["state_dict"])
    optimizer.load_state_dict(saved_model["optimizer"])


def plot(img, bboxes):
    i = np.array(img)
    h, w, _ = i.shape
    fig, ax = plt.subplots(1)
    ax.imshow(i)
    for bbox in bboxes:
        # bbox is on "cuda", transfer to cpu
        bbox = bbox.cpu().detach().numpy()
        bbox = bbox[2:]
        top_left_x = bbox[0] - bbox[2] / 2
        top_left_y = bbox[1] - bbox[3] / 2
        rect = patches.Rectangle((top_left_x * w, top_left_y * h),
                                 bbox[2] * w,
                                 bbox[3] * h,
                                 linewidth=1,
                                 edgecolor="r",
                                 facecolor="none")
        ax.add_patch(rect)
    plt.show()
