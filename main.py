import numpy as np

import tracker
from detector import Detector
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

red_line = []

# 判断是否在中心位置
def getCenterBox(bbox, w, h):
    bboxes = []
    for x in bbox:
        # x1, y1, x2, y2, lbl, conf = x[1] + x[3]
        p = (x[1] + x[3])/2
        if p<w/4*3 and p>w/4:
            bboxes.append(x)
    return bboxes

if __name__ == '__main__':

    video_path = 'videos\\2.mp4'
    output = 'result2.mp4' # 保存图像
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    # 获取视频帧的大小
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 720 
    frame_height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 1280
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 720 
    # frame_width= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 1280


    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((frame_width, frame_height), dtype=np.uint8)

    # 初始化2个撞线polygon
    # list_pts_blue = [640 720,  650 720, 650 0 ,640 0 ]
    # list_pts_blue = [[frame_width, frame_height/2],  [frame_width,frame_height/2 +10], [0, frame_height/2 +10], [0, frame_height/2]]
    # sp 竖着时
    list_pts_blue = [[frame_height,frame_width/2 ],  [frame_height, frame_width/2 + 10], [0, frame_width/2 + 10], [0, frame_width/2]]
    # sp 横着时
    # list_pts_blue = [[frame_height/2,frame_width],  [frame_height/2 + 20, frame_width], [frame_height/2 + 20, 0], [frame_height/2, 0]]


    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((frame_width, frame_height), dtype=np.uint8)
    # list_pts_yellow = [[0, 1020],  [1920, 1020], [1920, 1000], [0, 1000]]
    # (frame_width, frame_height)

    # list_pts_yellow = [[frame_width, frame_height/2],  [frame_width, frame_height/2 - 10], [0, frame_height/2 - 10], [0, frame_height/2]]
    list_pts_yellow = [[frame_height,frame_width/2],  [frame_height, frame_width/2 - 10], [0, frame_width/2 - 10], [0, frame_width/2]]
    # sp 横着时
    # list_pts_yellow = [[frame_height/2,frame_width],  [frame_height/2 - 20, frame_width], [frame_height/2 - 20, 0], [frame_height/2, 0]]

    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 设置检测的边框 我们选择为中间部分
    # 边界
    mask_image_temp = np.zeros((frame_width, frame_height), dtype=np.uint8)

    list_pts_red_1 = [[frame_height,(frame_width / 4) * 3],  [frame_height, frame_width / 4 * 3 + 5], [0, (frame_width/4*3) + 5], [0, (frame_width / 4) * 3]]
    list_pts_red_2 = [[frame_height,(frame_width / 4)],  [frame_height, frame_width / 4 + 5], [0, frame_width/4 + 5], [0, frame_width/4]]

    # list_pts_red_1 = [[frame_height/4*3,frame_width],  [frame_height / 4 * 3 + 10, frame_width], [frame_height/4*3+ 10, 0], [frame_height/4*3, 0]]
    # list_pts_red_2 = [[frame_height/4,frame_width],  [frame_height / 4 + 10, frame_width / 4 + 10], [frame_height/4+ 10, 0], [frame_height/4, 0]]

    ndarray_pts_red_1 = np.array(list_pts_red_1, np.int32)
    ndarray_pts_red_2 = np.array(list_pts_red_2, np.int32)

    polygon_red_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_red_1], color=1)
    mask_image_temp = np.zeros((frame_width, frame_height), dtype=np.uint8)
    polygon_red_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_red_2], color=1)

    polygon_red_value_1 = polygon_red_value_1[:, :, np.newaxis]
    polygon_red_value_2 = polygon_red_value_2[:, :, np.newaxis]

    red_color_plate = [0, 0, 255]
    red_image_1 = np.array(polygon_red_value_1 * red_color_plate, np.uint8)
    red_image_2 = np.array(polygon_red_value_2 * red_color_plate, np.uint8)


    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (frame_width, frame_height))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image + red_image_1 + red_image_2
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (frame_width, frame_height))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(frame_width * 0.01), int(frame_height * 0.05))

    # 初始化 yolov5
    # yaml = 'data\Tea\yolov8-tea.yaml'
    # weights = './weights/bestv8.pt'
    # yolo = True


    # 初始化yolov8
    yaml = 'data\Tea\yolov8-tea.yaml'
    weights = 'weights\\yolo_checkpoint\\best.pt'
    yolo = True
    half = False

    classify_id = ['1']
    # classify_id = ['Tea']
    

    # 以下代码是用于测试人群进行对代码的理解
    # weights = 'yolov5l.pt'
    # yolo = False
    # classify_id = ['person', 'bicycle', 'car', 'motorcycle']
    # video_path = '正东骑单车.mp4'


    # 修改Detector 25行进行加载yolov8
    # 如果使用ultralytics 进行训练 需要设置yolo为True 
    # 由于本代码为适应河南理工大学“华为杯”比赛，并未使用gpu以及half，后续可以将版本提高使用half以及gup加快运算
    detector = Detector(weights, classify_id, yaml, yolo=yolo, half=half)

    video_path = video_path
    # 打开视频
    capture = cv2.VideoCapture(video_path)
    
    fps = int(capture.get(5))
    print('fps:', fps)
    t = int(1000 / fps)

    # 当前照片中目标的数量
    current_nums = 0
    # 保存图像
    videoWriter = None

    while True:
        # 读取每帧图片
        _, im = capture.read()
        # 测试
        # im = cv2.imread('data22\\IMG_20210630_161133.jpg')
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        # im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)
        # 获取im中心box
        w, h, _ = im.shape
        # bboxes = getCenterBox(bboxes, w, h)
        current_nums = len(bboxes)
        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    pass

                    # 判断 黄polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 外出方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 外出+1
                        up_count += 1

                        print(f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')

                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_blue_polygon:
                        # 进入+1
                        down_count += 1

                        print(f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')

                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            pass
        pass

        text_draw = 'DOWN: ' + str(down_count) + \
                    ' , UP: ' + str(up_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)
        # 保存结果图像
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            fps = 30
            videoWriter = cv2.VideoWriter(
                output, fourcc, fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

        videoWriter.write(output_image_frame)

        cv2.imshow('demo', output_image_frame)
        # cv2.waitKey(1)
        cv2.waitKey(t)

        pass
    Tea_sum = current_nums + up_count
    print(f"当前影像中的签字数目为：{Tea_sum}")
    print(f"当前Track中的id数目为{tracker.deepsort.tracker._next_id - 1}")
    pass

    capture.release()
    cv2.destroyAllWindows()