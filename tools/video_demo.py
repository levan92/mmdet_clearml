# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import requests
import time

import cv2
import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument(
        '--inference-addr-det',
        default='detection_backend:8080',
        help='Address and port of the inference server for detection')
    parser.add_argument(
        '--inference-addr-clas',
        default='classification_backend:8083',
        help='Address and port of the inference server for classification')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    parser.add_argument(
        '--classification',
        action='store_true',
        help='Whether to do classification as 2nd step')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, ('Please specify at least one operation (save/show the video) with the argument "--out" or "--show"')

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

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        _ , encoded_image = cv2.imencode('.jpg', frame)
        det_response = requests.post(det_url, encoded_image.tobytes())
        det_results = det_response.json()
        # print(det_results)

        drawn_image = frame.copy()
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
                crop = frame[t:b, l:r]
                try:
                    _ , encoded_crop = cv2.imencode('.jpg', crop)
                except Exception as e:
                    print(f'error when encoding crop: {e}')
                    print(f'frame: {frame}')
                    print(f'crop dimensions: {l}, {t}, {r}, {b}')
                clas_response = requests.post(clas_url, encoded_crop.tobytes())
                clas_results = clas_response.json()
                classname = max(clas_results, key=clas_results.get)
                score = clas_results[classname]
                if classname == '5-Class4':
                    classname = 'Class4'
                
            text = f'{classname}: {score:.2f}'
            draw_color = bbox_colors[classname]

            cv2.rectangle(drawn_image, (l,t), (r,b), color=draw_color, thickness=2)
            cv2.putText(drawn_image, text, (l+5, b-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, draw_color, 1)

        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(drawn_image, 'video', args.wait_time)
        if args.out:
            video_writer.write(drawn_image)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
