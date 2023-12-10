Training data used is VOC2007 trainval and VOC2012 2012 train.

Set up process:
1. Download VOC2007 trainval, VOC2012 train. After unzipping, move folder VOC2007 and folder VOC2012 under a same folder, and name this folder "VOCdevkit". Make sure VOCdevkit is in the same folder as the rest of the YOLO codes.
2. run train.py, once the model get > 0.85 mAP, the parameters will be saved. To view the results, change LOAD_MODEL to True.
