from PIL import Image
import shutil
import pandas as pd 
import numpy as np
import yaml
import os
from sklearn.model_selection import KFold
from pathlib import Path
import wandb

def convert_bbox_to_yolo(bbox,img_width,img_height):
    if bbox == 0:
        return []
    string = bbox.replace('[', '').replace(']', '')
    string_list = string.split(',')
    int_list = np.array([int(float(x)) for x in string_list])
    x,y,w,h = int_list
    x_center = x + w / 2
    y_center = y + h / 2
    w_norm = w / img_width
    h_norm = h / img_height
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    return [x_center_norm, y_center_norm, w_norm, h_norm]


def convert_to_yolo_bboxes(image_ids,original_bboxes,images_path):
    bboxes = []
    for image_id, original_bbox in zip(image_ids,original_bboxes):
        img = Image.open(f"{images_path}/{image_id}.tif") 
        width = img.width 
        height = img.height
        yolo_bbox = convert_bbox_to_yolo(original_bbox,width,height)
        bboxes.append(yolo_bbox)
    
    return bboxes




def setup_experiment(dataframe, exp_type="fit", n_folds=None, seed=42):
    yamls = []
    
    yaml_file = "classes.yaml"
    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    
    parent_dir = Path(os.path.abspath('./experiment'))
    X = dataframe['image_id'].unique()

    if exp_type == "fit":
        img_target_path = parent_dir / "train" / "images"
        label_target_path = parent_dir / "train" / "labels"
        os.makedirs(img_target_path, exist_ok=True)
        os.makedirs(label_target_path, exist_ok=True)

        create_labels(X , img_target_path, dataframe)

        yaml_path = parent_dir / "train.yaml"
        yamls.append(str(yaml_path))
        with open(yaml_path, "w") as ds_y:
            yaml.safe_dump({
                "path": str(parent_dir),
                "train": "train",
                "val": "train",
                "names": classes,
            }, ds_y)

        return yamls
    
    else:
        train_images = [] 
        val_images = [] 
        cv = KFold(n_splits=n_folds,shuffle=True,random_state=seed)
        for i, (train_indices, val_indices) in enumerate(cv.split(X)):
            fold_train_images = X[train_indices]
            fold_val_images = X[val_indices]
            train_images.append(fold_train_images)
            val_images.append(fold_val_images)

            for split, images in zip(["train", "val"], [fold_train_images, fold_val_images]):
                img_target_path = parent_dir / f"fold_{i}" / split / "images"
                label_target_path = parent_dir / f"fold_{i}" / split / "labels"
                os.makedirs(img_target_path, exist_ok=True)
                os.makedirs(label_target_path, exist_ok=True)

                create_labels(images, img_target_path, dataframe)
                
            dataset_path = parent_dir / f"fold_{i}"
            dataset_yaml = dataset_path / "config.yaml"

            yamls.append(str(dataset_yaml))
            with open(dataset_yaml, "w") as ds_y:
                yaml.safe_dump({
                    "path": str(dataset_path),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                }, ds_y)

        return yamls, train_images, val_images

def create_labels(image_ids, images_target, dataframe):
    for image_id in image_ids:
        bboxes = dataframe.loc[dataframe['image_id'] == image_id]['bbox']
        labels = dataframe.loc[dataframe['image_id'] == image_id]['category_id']
        img_target_path = Path(images_target) / f"{image_id}.tif"
        label_target_path = img_target_path.with_suffix('.txt').as_posix().replace("images", "labels")
        
        if (labels == 0).all():
            continue
        else:
            with open(label_target_path, 'w') as f:
                for label, bbox in zip(labels, bboxes):
                    x, y, w, h = bbox
                    f.write(f'{label - 1} {x} {y} {w} {h}\n')

def copy_images(image_ids,images_src,images_target):
    for image_id in image_ids:
        img_src = images_src + f'/{image_id}.tif'
        img_target = images_target + f'/{image_id}.tif'
        shutil.copy(img_src,img_target)


def load_data(dataset_path,mode='fit'):
    if mode == 'fit':
        train_df = pd.read_csv(f"{dataset_path}/Train.csv")
        train_df.fillna(0,inplace=True)
        train_df['category_id'] = train_df['category_id'].map(int)
        train_df['bbox'] = convert_to_yolo_bboxes(train_df['image_id'],train_df['bbox'],dataset_path+"/Images")
        return train_df    
    else:
        ss = pd.read_csv(f"{dataset_path}/SampleSubmission.csv")
        test = pd.read_csv(f"{dataset_path}/Test.csv")
        return test, ss
    

def setup_logging(job_type='training'):
    wandb.login(key=os.environ['api_key'])
    wandb.init(project="arm-unicef", job_type=job_type)