import os
import cv2
import json
from PIL import Image
import argparse
from collections import defaultdict
import shapely.geometry as shgeo

# acceptable image suffixes
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def parse_args():
    parser = argparse.ArgumentParser(description='convert data to COCO format')
    parser.add_argument('--srcpath', help='folder of img, ann and labels.txt')
    parser.add_argument('--mode', help='train or test', default='train')
    parser.add_argument('--dstpath', help='store_path -- ../train.json')
    parser.add_argument('--classes', help='class_names (split by ,)', default=None)
    args = parser.parse_args()

    return args


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = ext is not None
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    f = open(filename, 'r')

    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            if len(splitlines) < 9:
                # handle bbox (x,y,w,h,class,difficult)
                if len(splitlines) == 6:
                    object_struct['bbox'] = [float(splitlines[0]), float(splitlines[1]),
                                             float(splitlines[2]), float(splitlines[3])]
                    object_struct['area'] = float(splitlines[2]) * float(splitlines[3])
                    object_struct['name'] = splitlines[4]
                    objects.append(object_struct)
                continue
            if len(splitlines) >= 9:
                    object_struct['name'] = splitlines[8]
            if len(splitlines) == 9:
                object_struct['difficult'] = '0'
            elif len(splitlines) >= 10:
                object_struct['difficult'] = splitlines[9]
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            objects.append(object_struct)
        else:
            break
    return objects


def TuplePoly2Poly(poly):
    return [poly[0][0], poly[0][1],
           poly[1][0], poly[1][1],
           poly[2][0], poly[2][1],
           poly[3][0], poly[3][1]]


def parse_dota_poly2(filename):
    """
        parse the dota ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    objects = parse_dota_poly(filename)
    for obj in objects:
        if 'poly' in obj:
            obj['poly'] = TuplePoly2Poly(obj['poly'])
            obj['poly'] = list(map(int, obj['poly']))
        elif 'bbox' in obj:
            obj['bbox'] = list(map(int, obj['bbox']))
    return objects


def DOTA2COCOTrain(srcpath, destfile, cls_names, coco_mode=1,
                   split_classes_dict=None,
                   ignore_classes=['plane', 'helicopter'], verbose=True):
    """
    :param srcpath: directory with images/ and labelTxt/
    :param destfile: file_name for the COCO JSON to be stored be; must be '.json'
    :param cls_names: class_names for training; the class_index is also used for inference;
    :param coco_mode: 0 for normal class_name indexing;
                 1 for 1 Vehicle class,
                 2 for split_classes_dict
                 (i.e split_classes_dict = {'vehicle': ['small-vehicle', 'large-vehicle'],
                                            'helicopter': ['helicopter'],
                                            'plane': ['plane']}
    :param split_classes_dict: ^ read above
    :param ignore_classes: classes to be ignored
    :param verbose: True
    """

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    classes_counter = defaultdict(lambda: 0)
    if verbose:
        print("Counting Classes")

    data_dict = {'images': [], 'categories': [], 'annotations': []}
    if coco_mode == 0:
        for idex, name in enumerate(cls_names):
            single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
            data_dict['categories'].append(single_cat)
    elif coco_mode == 1:
        data_dict['categories'].append({'id': 1, 'name': 'Vehicle', 'supercategory': 'Vehicle'})
    else:
        assert split_classes_dict is not None
        for idex, name in enumerate(split_classes_dict.keys()):
            single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
            data_dict['categories'].append(single_cat)
            class_distributor = {}
            for key, list_of_values in split_classes_dict.items():
                for val in list_of_values:
                    class_distributor[val] = key

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = GetFileFromThisRootDir(labelparent)
        for file in filenames:
            imagepath = False
            for img_format in IMG_FORMATS:
                basename = f'{custombasename(file)}.{img_format}'
                test_imagepath = os.path.join(imageparent, basename)
                if os.path.isfile(test_imagepath):
                    imagepath = test_imagepath
                    break

            if imagepath:
                img = cv2.imread(imagepath)
                height, width, c = img.shape

                single_image = {'file_name': basename, 'id': image_id, 'width': width, 'height': height}
                data_dict['images'].append(single_image)

                # annotations
                objects = parse_dota_poly2(file)
                for obj in objects:
                    if obj['name'] in ignore_classes:
                        continue
                    single_obj = {'area': obj['area'], 'iscrowd': 0}
                    if coco_mode == 0:
                        single_obj['category_id'] = cls_names.index(obj['name']) + 1
                    elif coco_mode == 1:
                        single_obj['category_id'] = 1
                    else:
                        obj['name'] = class_distributor[obj['name']]
                        single_obj['category_id'] = cls_names.index(obj['name']) + 1

                    # handle bbox
                    if 'bbox' in obj.keys():
                        single_obj['bbox'] = obj['bbox']
                        print('WARNING: COCO has HBB; No OBB. Do not train OBB')
                    # handle OBB
                    else:
                        single_obj['segmentation'] = [obj['poly']]
                        xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                                 max(obj['poly'][0::2]), max(obj['poly'][1::2])
                        width, height = xmax - xmin, ymax - ymin
                        single_obj['bbox'] = xmin, ymin, width, height
                    single_obj['image_id'] = image_id
                    if width * height <= 0:
                        print("Skipping annotations that are too small: width / height == 0")
                        continue
                    data_dict['annotations'].append(single_obj)
                    single_obj['id'] = inst_count
                    inst_count = inst_count + 1

                    if verbose:
                        classes_counter[obj['name']] += 1
                image_id = image_id + 1

        if verbose:
            print(classes_counter)
            print("Dumping json now")
        json.dump(data_dict, f_out)


def DOTA2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = GetFileFromThisRootDir(imageparent)
        for file in filenames:
            _, img_format = os.path.splitext(file)
            if img_format.lower() not in IMG_FORMATS:
                print(f"Skipping {file} as it does not fit IMG_FORMATS")
                continue
            basename = custombasename(file) + img_format
            imagepath = os.path.join(imageparent, basename)

            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {'file_name': basename, 'id': image_id, 'width': width, 'height': height}
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    mode = args.mode.lower()

    class_names = os.environ.get('CLASSES', args.classes)
    class_names = class_names.split(',')

    if mode == 'train':
        DOTA2COCOTrain(srcpath, dstpath, class_names)
    else:
        DOTA2COCOTest(srcpath, dstpath, class_names)