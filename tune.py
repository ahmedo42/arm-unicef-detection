from ultralytics import YOLO 
from data_utils import copy_images,load_data,setup_experiment,setup_logging
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--dataset_path", type=str, default='./data')
parser.add_argument("--model_name", type=str, default='yolov8n.pt')
parser.add_argument("--project", type=str, default='tune')
args = parser.parse_args()


def tune():
    config = vars(args)
    train_df = load_data(config['dataset_path'])
    model = YOLO(config['model_name'])
    images_src =  config['dataset_path'] +"/Images/"
    images_train_target = f"./experiment/tune/images"
    yamls = setup_experiment(train_df,exp_type='fit',seed=config['seed'])
    copy_images(train_df['image_id'].unique(),images_src,images_train_target)

    model.tune(
        data = yamls[0],
        project = config['project'],
        epochs = config['epochs'],
        iterations = config['iterations'],
        batch = config['batch'],
        seed = config['seed'],
        plots=False,
        val=False,
    )

if __name__ == "__main__":
    tune()