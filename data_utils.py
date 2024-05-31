from PIL import Image
import shutil
import pandas as pd 
import numpy as np
import yaml
import os
from sklearn.model_selection import KFold
from pathlib import Path

EXP_TYPES = ["FIT","KFOLD"]
# FIT: fit the model to the whole training set 
# KFOLD: kfold cv

def convert_bbox_to_yolo(bbox,img_width,img_height):
    # converts the default x,y,w,h labelling scheme to yolo compatible labelling
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




def setup_experiment(dataframe, exp_type="FIT", n_folds=None):
    # Creates appropriate directory structure, labels, and YAML files based on experiment type
    yamls = []
    
    # Load class names from the YAML file
    yaml_file = "classes.yaml"
    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    
    # Set up parent directory
    parent_dir = Path(os.path.abspath('./experiment'))
    X = dataframe['image_id'].unique()

    if exp_type == "FIT":
        # Set up directories for FIT experiment
        img_target_path = parent_dir / "train" / "images"
        label_target_path = parent_dir / "train" / "labels"
        os.makedirs(img_target_path, exist_ok=True)
        os.makedirs(label_target_path, exist_ok=True)

        # Create datasets for training
        create_labels(X , img_target_path, dataframe)

        # Create YAML configuration for FIT experiment
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
        # Set up directories and configurations for cross-validation experiment
        train_images = [] 
        val_images = [] 
        cv = KFold(n_splits=n_folds)
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

                # Create datasets for training and validation
                create_labels(images, img_target_path, dataframe)
                
                # Create YAML configuration for each fold and split
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
            print(f"image {image_id} is empty")
            open(label_target_path, 'a').close()
        else:
            with open(label_target_path, 'w') as f:
                print(f"writing labels for image {image_id}")
                for label, bbox in zip(labels, bboxes):
                    x, y, w, h = bbox
                    f.write(f'{label} {x} {y} {w} {h}\n')

def move_images(image_ids,images_src,images_target):
    for image_id in image_ids:
        images_src = (images_src / image_id + '.tif')
        images_target = (images_target / image_id + '.tif')
        shutil.move(images_src,images_target)

def cleanup(required_path,destination_path):
    # deletes directory of a fold and restores data to the original parent directory
    # useful for "undoing" cross-validation 
    os.makedirs(destination_path, exist_ok=True)
    for root, _, files in os.walk(required_path):
        for file in files:
            if file.lower().endswith('.tif'):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(destination_path, file)
                
                shutil.move(source_file_path, target_file_path)
    
    shutil.rmtree(required_path)



def create_img_label_mapping(df):

    images = df['image_id'].unique()
    image_ids = []
    labels = []
    for image_id in images:
        categories = df.loc[df['image_id'] == image_id]['category_id']
        label =  ''.join(sorted([str(l) for l in categories.unique()]))
        image_ids.append(image_id)
        labels.append(label)

    labels_df = pd.DataFrame(
        {
            "image_id" : image_ids,
            "label": labels,
        }
    )
    return labels_df
