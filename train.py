import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from dataset import Dataset
from loss import Loss
from architecture import YOLO
from config import *


class Augment(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for transform in self.transforms:
            img, bboxes = transform(img), bboxes
        return img, bboxes


# define the transformation to be a resize operation
transforms = Augment([transforms.Resize((448, 448)), transforms.ToTensor()])


# training procedure
def train(dataloader, model, optim, loss_fn):
    # progress bar set up
    dataloader = tqdm(dataloader, leave=True)
    average_loss = []

    for idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = loss_fn(output, y)
        average_loss.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

        dataloader.set_postfix(loss=loss.item())
    print(f"average loss: {sum(average_loss) / len(average_loss)}")


def main():
    model = YOLO().to(DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    loss = Loss()
    if LOAD_MODEL:
        load(torch.load(LOAD_MODEL_FILE), model, optimizer)
    train_dataset = Dataset("train.csv", transforms=transforms, img_dir=IMG_DIR, tar_dir=TAR_DIR)
    #test_dataset = Dataset("VOCdevkit/tar/test.csv", transforms=transforms, img_dir=IMG_DIR, tar_dir=TAR_DIR)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=2,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)
    #test_loader = DataLoader(dataset=test_dataset,
    #                         batch_size=BATCH_SIZE,
    #                         num_workers=2,
    #                         pin_memory=True,
    #                         shuffle=True,
    #                         drop_last=True)
    best_mAP = 0
    for epoch in range(EPOCHS):
        if LOAD_MODEL:
            model.eval()
            with torch.no_grad():
                for idx, (x, y) in enumerate(train_loader):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    bboxes, _ = bbox_cell2img(model(x), torch.Tensor([]))
                    for bbox_idx in range(BATCH_SIZE):
                        bbox = non_max_suppression(bboxes[bbox_idx], IOU_THRESHOLD, PROB_THRESHOLD)
                        plot(x[bbox_idx].permute(1, 2, 0).to("cpu"), bbox)
                    pred_bboxes, tar_bboxes = get_bboxes(train_loader, model, IOU_THRESHOLD, PROB_THRESHOLD)
                    mAP = mean_average_precision(pred_bboxes, tar_bboxes, IOU_THRESHOLD)
                    print(f"mAP of current model: {mAP}\n")
                    import sys
                    sys.exit()
        train(train_loader, model, optimizer, loss)
        pred_bboxes, tar_bboxes = get_bboxes(train_loader, model, IOU_THRESHOLD, PROB_THRESHOLD)
        mAP = mean_average_precision(pred_bboxes, tar_bboxes, IOU_THRESHOLD)
        print(f"Epoch: {epoch} Train mAP: {mAP}\n")
        if mAP > best_mAP and mAP > 0.5:
            best_mAP = mAP
            save({"state_dict": model.state_dict(),
                  "optimizer": optimizer.state_dict()}, filename=LOAD_MODEL_FILE)


if __name__ == "__main__":
    main()
