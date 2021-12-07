import os
import shutil
from .DOTA2COCO import DOTA2COCOTrain
import argparse


def retrieve_folders(video_path, scale_list, output_path, class_names,
                     coco_mode=0, split_classes_dict=None):

    output_img_path = os.path.join(output_path, 'images')
    output_txt_path = os.path.join(output_path, 'labelTxt')
    for video_dir in video_path:
        for scale_type in os.listdir(video_dir):
            if scale_type in scale_list:
                scaled_img_dir = os.path.join(video_dir, scale_type, 'images')
                scaled_txt_dir = os.path.join(video_dir, scale_type, 'labelTxt')
                for image in os.listdir(scaled_img_dir):
                    shutil.move(os.path.join(scaled_img_dir, image), os.path.join(output_img_path, image))
                for txt in os.listdir(scaled_txt_dir):
                    shutil.move(os.path.join(scaled_txt_dir, txt), os.path.join(output_txt_path, txt))
    if 'train' in output_img_path:
        json_path = 'train.json'
    elif 'val' in output_img_path:
        json_path = 'val.json'
    else:
        json_path = 'test.json'
    DOTA2COCOTrain(output_path, json_path, class_names,
                             coco_mode=coco_mode, split_classes_dict=split_classes_dict)
    shutil.move(json_path, os.path.join(output_path, json_path))


def merge_videos(train_folders=None, val_folders=None, test_folders=None,
                 class_names=None, mode='train',
                 coco_mode=0, split_classes_dict=None):
    """
    :param train_folders: videos that are for training
    :param val_folders: videos that are for val
    :param test_folders: videos that are for test
    :param class_names: class names
    :param mode: train or test
    :param coco_mode: 0 for normal class_name indexing;
                 1 for 1 Vehicle class,
                 2 for split_classes_dict
                 (i.e split_classes_dict = {'vehicle': ['small-vehicle', 'large-vehicle'],
                                            'helicopter': ['helicopter'],
                                            'plane': ['plane']}
    :param split_classes_dict: ^ read above
    """

    training_scale = ('0.5', '1', '1.5')
    val_scale = ('0.5', '1', '1.5')
    test_scale = ('1',)

    train_path = 'datasets/train/'
    val_path = 'datasets/val/'
    test_path = 'datasets/test/'

    if mode == 'train':
        if type(train_folders) not in [list, tuple]:
            train_folders = train_folders.split(',')
        if type(val_folders) != list:
            val_folders = val_folders.split(',')
        for path_dir in [train_path, val_path]:
            os.makedirs(os.path.join(path_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(path_dir, 'labelTxt'), exist_ok=True)

        retrieve_folders(train_folders, training_scale, train_path, class_names=class_names,
                         coco_mode=coco_mode, split_classes_dict=split_classes_dict)
        retrieve_folders(val_folders, val_scale, val_path, class_names=class_names,
                         coco_mode=coco_mode, split_classes_dict=split_classes_dict)

    else:
        if type(test_folders) not in [list, tuple]:
            test_folders = test_folders.split(',')
        os.makedirs(os.path.join(test_path, 'images'))
        os.makedirs(os.path.join(test_path, 'labelTxt'))
        retrieve_folders(test_folders, test_scale, test_path, class_names=class_names,
                         coco_mode=coco_mode, split_classes_dict=split_classes_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge videos for Train, Val and Test')
    parser.add_argument('--train_folders', default=None)
    parser.add_argument('--val_folders', default=None)
    parser.add_argument('--test_folders', default=None)
    parser.add_argument('--mode', help='train or test (train for S3, test for mAP eval locally)', default='train')
    parser.add_argument('--classes', help='class_names (split by ,)', default=None)
    parser.add_argument('--coco_mode', default=0)
    args = parser.parse_args()
    merge_videos(train_folders=args.train_folders,
                 val_folders=args.val_folders,
                 test_folders=args.test_folders,
                 class_names=args.classes,
                 coco_mode=args.coco_mode,
                 mode=args.mode)
