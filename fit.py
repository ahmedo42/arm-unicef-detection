from ultralytics import YOLO 
from data_utils import copy_images,load_data,setup_experiment,setup_logging
import argparse
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
parser.add_argument("--project", type=str, default='fitting')
parser.add_argument("--lr0", type=float, default= 1e-3)
parser.add_argument("--optimizer", type=str, default= 'Adam')
parser.add_argument("--flipud", type=float, default= 0.0)

args = parser.parse_args()


def fit():
    config = vars(args)
    train_df = load_data(config['dataset_path'])
    setup_logging(job_type='training')
    model = YOLO(config['model_name'])
    images_src =  config['dataset_path'] +"/Images/"
    images_train_target = f"./experiment/train/images"
    yamls = setup_experiment(train_df,exp_type='fit',seed=config['seed'])
    copy_images(train_df['image_id'].unique(),images_src,images_train_target)
    close_mosaic = config['epochs'] // 10


    model.train(
        data = yamls[0],
        task = 'detect',
        project = config['project'],
        epochs = config['epochs'],
        imgsz = config['imgsz'],
        box = config['box'],
        cls = config['cls'],
        batch = config['batch'],
        dropout = config['dropout'],
        seed = config['seed'],
        plots=False,
        close_mosaic = close_mosaic,
        optimizer = config['optimizer'],
        lr0 = config['lr0'],
        warmup_epochs=0.3,
        flipud=config['flipud']
    )

if __name__ == "__main__":
    fit()