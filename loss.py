import torch
import torch.nn as nn
from utils import iou


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, tar):
        #  The prediction is in form of a vector with length 7*7*30
        #  reshape first
        pred = pred.reshape(-1, 7, 7, 30)

        lambda_coord = 5
        lambda_noobj = .5

        # first part of the loss: the coordinates errors
        # calculate IOUs to determine which bounding box is responsible
        iou1 = iou(pred[..., 21:25], tar[..., 21:25])
        iou2 = iou(pred[..., 26:30], tar[..., 21:25])
        ious = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0)
        iou_maxes, betterbox = torch.max(ious, dim=0)
        betterbox = betterbox.unsqueeze(-1)
        existed_obj = tar[..., 20].unsqueeze(3) # Iobj_i in paper
        bbox_pred = existed_obj * (betterbox * pred[..., 26:30] +
                                   (1 - betterbox) * pred[..., 21:25])
        bbox_pred[..., 2:4] = torch.sign(bbox_pred[..., 2:4]) * torch.sqrt(
            torch.abs(bbox_pred[..., 2:4] + 1e-6)
        )
        bbox_tar = existed_obj * tar[..., 21:25]
        bbox_tar[..., 2:4] = torch.sqrt(bbox_tar[..., 2:4])
        # (N,7,7,4) -> (N*7*7,4)
        loss1 = lambda_coord * self.mse(torch.flatten(bbox_pred, end_dim=-2), torch.flatten(bbox_tar, end_dim=-2))

        # second part of the loss: the object loss
        guessed_existed_obj = betterbox * pred[..., 25].unsqueeze(-1) + (1 - betterbox) * pred[..., 20].unsqueeze(-1)
        # (N,7,7,1) -> (N*7*7,)

        loss2 = self.mse(torch.flatten(existed_obj * guessed_existed_obj),
                         torch.flatten(existed_obj * tar[..., 20].unsqueeze(-1)))   # 增加这个loss增加的是bbox的个数？

        # third part of the loss: the no object loss
        # (N,7,7,1) -> (N*7*7,)
        loss3 = lambda_noobj * (
            self.mse(torch.flatten((1 - existed_obj) * pred[..., 20].unsqueeze(-1), start_dim=1),
                     torch.flatten((1 - existed_obj) * tar[..., 20].unsqueeze(-1), start_dim=1))
            +
            self.mse(torch.flatten((1 - existed_obj) * pred[..., 25].unsqueeze(-1), start_dim=1),
                     torch.flatten((1 - existed_obj) * tar[..., 20].unsqueeze(-1), start_dim=1))
        )

        # fourth part of the loss: the class loss
        prob_pred = existed_obj * pred[..., :20]
        prob_tar = existed_obj * tar[..., :20]
        # (N,7,7,20) -> (N*7*7,20)
        loss4 = self.mse(torch.flatten(prob_pred, end_dim=-2), torch.flatten(prob_tar, end_dim=-2))

        loss = loss1 + loss2 + loss3 + loss4
        return loss
