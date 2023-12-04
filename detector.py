import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from ultralytics import YOLO

class Detector:

    def __init__(self, weights, classify_id, yaml, yolo=False, half=False):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1
        self.yaml = yaml
        self.weights = weights 

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        # yolov5 加载权重
        

        # model = YOLO(self.weights)
        self.half = half
        self.yolo = yolo

        if self.yolo:
            try:
                from ultralytics import YOLO 
                model = YOLO(self.weights)
            except ImportError:
                print('导出ultralytics库失败')
                model = attempt_load(self.weights, device=self.device)
                if self.device != 'cpu':
                    model.to(self.device).eval()
                    model.half()
                else:
                    model.to(self.device).eval()
        else:
            model = attempt_load(self.weights, device=self.device)
            model.to(self.device).eval()
            # 由于版本底不能使用half
            if self.half:
                model.half()    

    
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names
        
        self.classify_id = classify_id

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        if self.half:
            img = img.half()
        else:
            img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)
        # 最初代码
        # 对下面作部分修改，便于使用yolov8
        if self.yolo:
            results = self.m(img, conf=0.25, iou=0.4, half=self.half, device=self.device, verbose=False)
            pred = [results[0].boxes.data]
            self.names[0] = '1'
        else:
            pred = self.m(img, augment=False)[0] # 返回tensor 1*15120*7
            pred = pred.float()
            pred = non_max_suppression(pred, self.threshold, 0.4)

         


        boxes = []  
        for det in pred:

            if det is not None and len(det):
                det1 = det.clone()
                det1[:, :4] = scale_coords(
                    img.shape[2:], det1[:, :4], im0.shape).round() # 384, 640  

                for *x, conf, cls_id in det1:
                    lbl = self.names[int(cls_id)]
                    # 在这进行修改
                    if lbl not in self.classify_id:
                        continue
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes
