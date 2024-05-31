from PIL import Image
import shutil
import pandas as pd 
import numpy as np
import yaml
import os
from sklearn.model_selection import KFold

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


def convert_to_yolo_bboxes(image_ids,original_bboxes):
    bboxes = []
    for image_id, original_bbox in zip(image_ids,original_bboxes):
        img = Image.open(f"./data/Images/{image_id}.tif") 
        width = img.width 
        height = img.height
        yolo_bbox = convert_bbox_to_yolo(original_bbox,width,height)
        bboxes.append(yolo_bbox)
    
    return bboxes



def setup_experiment(dataframe,images_path,exp_type="FIT",n_folds=None):
    # creates appropriate directory strucutre, labels and YAML files based on experiment type
    yamls = []

    yaml_file = "classes.yaml"
    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]

    parent_dir = './experiment'
    X = dataframe['image_id'].unique()

    if exp_type == "FIT":

        img_target_path = parent_dir + "/train" + "/images"
        label_target_path = parent_dir+ "/train" + "/labels"
        os.makedirs(img_target_path,exist_ok=True)
        os.makedirs(label_target_path,exist_ok=True)
        create_datasets(images,images_path,img_target_path,dataframe)
        split_path = parent_dir + "/train"
        yaml_path = split_path + "/train.yaml"
        yamls.append(yaml_path)
        with open(yaml_path, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_path.as_posix(),
                    "train": "train",
                    "val": "train",
                    "names": classes,
                },
                ds_y,
            )
    else:

        cv = KFold(n_splits=n_folds)
        for i,(train_indices,val_indices) in enumerate(cv.split(X)):

            train_images = X.loc[train_indices]
            val_images = X.loc[val_indices]

            for split in ["train","val"]:
                img_target_path = parent_dir + f"/fold_{i}" + f"/{split}" + "/images"
                label_target_path = parent_dir + f"/fold_{i}" + f"/{split}" + "/labels"
                os.makedirs(img_target_path,exist_ok=True)
                os.makedirs(label_target_path,exist_ok=True)

                if split == "train":
                    images = train_images
                else:
                    images = val_images

                create_datasets(images,images_path,img_target_path,dataframe)
                split_path = parent_dir + f"fold_{i}" + f"/{split}"

                dataset_yaml = parent_dir + f"/fold_{i}" + f"/{split}" + f"{split}.yaml"
                yamls.append(dataset_yaml)

                with open(dataset_yaml, "w") as ds_y:
                    yaml.safe_dump(
                        {
                            "path": split_path.as_posix(),
                            "train": "train",
                            "val": "val",
                            "names": classes,
                        },
                        ds_y,
                    )
        return yamls

def create_datasets(image_ids,images_src,images_target,dataframe):
    for image_id in image_ids:
        bboxes = dataframe.loc[dataframe['image_id'] == image_id]['bbox']
        labels = dataframe.loc[dataframe['image_id'] == image_id]['category_id']
        img_original_path = f"{images_src}/{image_id}.tif"
        img_target_path = f"{images_target}/{image_id}.tif"
        label_target_path = img_target_path.replace("images,labels").replace(".tif",".txt")
        if (labels == 0).all():
            print(f"image {image_id} is empty")
            open(label_target_path, 'a').close()
        else:

            with open(label_target_path,'w') as f:
                print(f"writing labels for image {image_id}")
                for label,bbox in zip(labels,bboxes):
                    x, y, w, h = bbox
                    f.write(f'{label} {x} {y} {w} {h}' + '\n')
            print(f"copying image from {img_original_path} to {img_target_path}")
            shutil.copy(img_original_path,img_target_path) 



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
