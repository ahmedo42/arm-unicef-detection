import collections
import os
from data_utils import load_data
from ultralytics import YOLO
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--conf", type=float, default=0.5)
parser.add_argument("--iou", type=float, default=0.7)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--dataset_path", type=str, default='./data')
parser.add_argument("--model_path", type=str, default='best.pt')
args = parser.parse_args()

def predict():
    config = vars(args)
    test, ss = load_data(config.dataset_path)
    model = YOLO(config.model_path)
    image_paths = []
    for image_id in test['image_id']:
        image_paths.append(os.path.join(config.dataset_path,image_id+'.tif'))

    for img_idx in range(0,len(image_paths),config.batch):
        img_path_subset = image_paths[img_idx : img_idx + config.batch]
        preds = model.predict(img_path_subset,conf=config.conf , iou = config.iou)
        for pred,img_path in zip(preds,img_path_subset):
            classes = pred.boxes.cls.cpu().tolist()
            # increment classes by 1 to match competition mapping
            classes = collections.Counter([c+1 for c in classes])
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            for cls,count in classes.items():
                composite_name = image_id + '_' + str(int(cls))
                ss.loc[ss['image_id'] == composite_name,'Target'] = count 
    return ss 

if __name__ == '__main__':
    predict()