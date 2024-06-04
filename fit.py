from ultralytics import YOLO 
from data_utils import copy_images,load_data,setup_experiment
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


args = parser.parse_args()


def fit():
    config = vars(args)
    train_df = load_data(config.dataset_path)
    model = YOLO(config.model_name,)
    images_src =  config.data +"/Images/"
    images_train_target = f"./experiment/train/images"
    copy_images(train_df['image_id'].unique(),images_src,images_train_target)
    yamls = setup_experiment(train_df,exp_type='fit',seed=config.seed)
    close_mosiac = config.epochs // 10

    results = model.train(
        data = yamls[0],
        project = config.project,
        epochs = config.epochs,
        imgsz = config.imgsz,
        box = config.box,
        cls = config.cls,
        batch = config.batch,
        dropout = config.dropout,
        seed = config.seed,
        plots=False,
        val=False,
        close_mosiac = close_mosiac,
        optimizer = config.optimizer,
        lr0 = config.lr0,
    )
    return results


if __name__ == "__main__":
    fit()