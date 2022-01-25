import requests
import time
from argparse import ArgumentParser

import cv2
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image file')
    parser.add_argument(
        '--inference-addr-det',
        default='detection_backend:8080',
        help='Address and port of the inference server for detection')
    parser.add_argument(
        '--inference-addr-clas',
        default='classification_backend:8083',
        help='Address and port of the inference server for classification')
    parser.add_argument(
        '--classification',
        action='store_true',
        help='Whether to do classification as 2nd step')
    args = parser.parse_args()
    return args


def main(args):
    det_url = 'http://' + args.inference_addr_det + '/predictions/detection'

    test_img = np.zeros((10,10,3))
    _ , encoded_test_image = cv2.imencode('.jpg', test_img)
    test_img_bytes = encoded_test_image.tobytes()
    # make sure detection torchserve container has started
    connection_succeed = False
    while not connection_succeed:
        try:
            # if successful, response should be empty list
            response = requests.post(det_url, test_img_bytes).json()
            if not response:
                connection_succeed = True
            else:
                time.sleep(2)
        except requests.exceptions.ConnectionError:
            time.sleep(2)
    print('DETECTION BACKEND STARTED')

    if args.classification:
        clas_url = 'http://' + args.inference_addr_clas + '/predictions/classification'

        # make sure classification torchserve container has started
        connection_succeed = False
        while not connection_succeed:
            try:
                # if successful, response should be dict of results
                response = requests.post(clas_url, test_img_bytes).json()
                if response.get('code', None) != 404:
                    connection_succeed = True
                else:
                    time.sleep(2)
            except requests.exceptions.ConnectionError:
                time.sleep(2)
        print('CLASSIFICATION BACKEND STARTED')

    bbox_colors = {'Class1': (0,0,255), 'Class2': (255,0,0), 'Class3': (255,157,0), 'Class4': (157,255,0), 'Class5': (0,255,255)}

    with open(args.img_path, 'rb') as image:
        det_response = requests.post(det_url, image)
    det_results = det_response.json()
    # print(det_results)

    img = cv2.imread(args.img_path)
    drawn_image = img.copy()
    for det_res in det_results:
        classname = det_res['class_name']
        score = det_res['score']
        l,t,r,b = det_res['bbox']
        l = int(l)
        t = int(t)
        r = int(r)
        b = int(b)

        # do classification on the bbox
        if args.classification:
            crop = img[t:b, l:r]
            _ , encoded_crop = cv2.imencode('.jpg', crop)
            clas_response = requests.post(clas_url, encoded_crop.tobytes())
            clas_results = clas_response.json()
            # print(f'clas_results: {clas_results}')
            classname = max(clas_results, key=clas_results.get)
            score = clas_results[classname]
            if classname == '5-Class4':
                classname = 'Class4'

        text = f'{classname}: {score:.2f}'
        draw_color = bbox_colors[classname]

        cv2.rectangle(drawn_image, (l,t), (r,b), color=draw_color, thickness=2)
        cv2.putText(drawn_image, text, (l+5, b-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, draw_color, 1)

    cv2.imwrite('test.jpg', drawn_image)


if __name__ == '__main__':
    args = parse_args()
    main(args)
