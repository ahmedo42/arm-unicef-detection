from ultralytics import YOLO 
from data_utils import copy_images,load_data,setup_experiment,setup_logging
import argparse
import shutil
import math 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cls", type=float, default=0.5)
parser.add_argument("--box", type=float, default=7.5)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--dataset_path", type=str, default='./data')
parser.add_argument("--model_name", type=str, default='yolov8n.pt')
parser.add_argument("--project", type=str, default='evaluation')
parser.add_argument("--n_splits",type=int,default=5)
parser.add_argument("--lr0", type=float, default= 1e-3)
parser.add_argument("--optimizer", type=str, default= 'Adam')

args = parser.parse_args()


def evaluate():
    config = vars(args)
    train_df = load_data(config['datset_path'])
    setup_logging(job_type='eval')
    results = {} 
    for i in range(config['n_splits']):
        model = YOLO(config['model_name'])
        images_src =  config['data'] +"/Images/"
        images_train_target = f"./experiment/fold_{i}/train/images"
        images_val_target = f"./experiment/fold_{i}/val/images"
        yamls,train_images,val_images = setup_experiment(train_df,exp_type='eval',seed=config['seed'])
        copy_images(train_images,images_src,images_train_target)
        copy_images(val_images,images_src,images_val_target)
        close_mosaic = config['epochs'] // 10
        warmup_epochs = math.ceil(config['epochs'] // 100)
        results[i] = model.train(
            data = yamls[i],
            project = config['project'],
            name = f'fold_{i}',
            epochs = config['epochs'],
            imgsz = config['imgsz'],
            box = config['box'],
            cls = config['cls'],
            batch = config['batch'],
            dropout = config['dropout'],
            seed = config['seed '],
            close_mosaic = close_mosaic,
            warmup_epochs = warmup_epochs
        )
        shutil.rmtree(f"./experiment/fold_{i}")

if __name__ == "__main__":
    evaluate()