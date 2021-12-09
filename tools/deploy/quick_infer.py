from uuid import RESERVED_MICROSOFT
from mmdet.apis import inference_detector, init_detector

def init(config, checkpoint, device='cuda:0'):
    return init_detector(config, checkpoint, device)

def infer(model, img):
    return inference_detector(model, img)

def postprocess(res, model, thresh=0.5):
    # Format output following the example ObjectDetectionHandler format
    out = []
    for class_index, class_result in enumerate(res):
        class_name = model.CLASSES[class_index]
        for bbox in class_result:
            bbox_coords = bbox[:-1].tolist()
            score = float(bbox[-1])
            if score >= thresh:
                out.append({
                    'class_name': class_name,
                    'bbox': bbox_coords,
                    'score': score
                })

    return out

if __name__ == '__main__':
    import cv2

    config = '../../configs/coco_mini/coco_mini_person-bicycle-car_s3_direct.py'
    checkpoint = '../weights/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117-6eda6d92.pth'
    imgpath = '/home/levan/Pictures/road.jpg'

    img = cv2.imread(imgpath)

    detector = init(config, checkpoint)
    res = infer(detector, img)
    results = postprocess(res, detector)

    img_show = img.copy()
    for res in results:
        l,t,r,b = res['bbox']
        cv2.rectangle(img_show, (int(l),int(t)), (int(r),int(b)), color=(255,255,0), thickness=1 )
    
    cv2.imshow('', img_show)
    cv2.waitKey(0)