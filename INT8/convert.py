import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='coco128',type=str, help="root path of images and labels, include images and labels and classes.txt")
parser.add_argument('--save_path', type=str,default='output.json', help="if not split the dataset, give a path to a json file")
arg = parser.parse_args()

def train_test_val_split(img_paths, ratio_train = 0.8, ratio_test = 0.1, ratio_val = 0.1):
    # here can modify the ratio of dataset division
    assert int(ratio_train + ratio_test + ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths, test_size = 1 - ratio_train, random_state = 233)
    ratio = ratio_val / (1 - ratio_train)
    val_img, test_img = train_test_split(middle_img, test_size = ratio, random_state = 233)
    print("nums of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img


def yolo2coco(root_path):
    originLabelsDir = os.path.join(root_path, 'labels')                                        
    originImagesDir = os.path.join(root_path, 'images')
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()
    # images dir name
    indexes = os.listdir(originImagesDir)


    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # labeled id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # support .png & .jpg format images
        txtFile = index.replace('images', 'txt').replace('.jpg', '.txt').replace('.png', '.txt')
        # read the width and height of the image
        im = cv2.imread(os.path.join(root_path, 'images/') + index)
        height, width, _ = im.shape

        # add image information
        dataset['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # if there is no label, skip it and only keep the image information
            continue
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x, y, w, h to x1, y1, x2, y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # the label number starts from 0
                cls_id = int(label[0])   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, the rectangle is the four vertices clockwise from the upper left corner
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # save result
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)

    json_name = os.path.join(root_path, 'annotations/{}'.format(arg.save_path))
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(json_name))

if __name__ == "__main__":
    root_path = arg.root_dir
    assert os.path.exists(root_path)
    print("Loading data from ", root_path)
    yolo2coco(root_path)
