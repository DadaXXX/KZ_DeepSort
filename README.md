# yolov8串串香视频竹签数量检测
本代码已经上传到github，可以通过访问https://github.com/DadaXXX/KZ.git  
## 1. 创建python环境
conda create -n KZ python=3.7  
cobda activate KZ
## 2. 安装相应的库
pip install -r requriements.txt
## 3. 下载代码以及相关数据
### 3.1 代码下载
git clone https://github.com/DadaXXX/KZ.git   
cd kz  
### 3.2 权重文件下载
由于github限制上传以及时间成本等原因，我们将训练的权重文件放在百度网盘上面供大家下载  

链接：https://pan.baidu.com/s/11dclp60dsb9bYX5FYxTfZA?pwd=6uhp 
提取码：6uhp

下载之后将 ckpt.t7文件 放在\weights\deepsort_checkpoint下  
下载之后将 kz_best.pt文件 放在\weights\yolo_checkpoint下 

## 4. 运行监测
运行监测代码在main.py\main_yolo.ipynb 中  
首先调整list_pts_blue以及list_pts_yellow设置监测框  

```python
list_pts_blue = [[1020, 1080],  [1000, 1080], [1000, 0], [1020, 0]]
list_pts_yellow = [[1000, 1080],  [980, 1080], [980, 0], [1000, 0]]
``` 
通过设置yaml、weights以及yolo进行不同权重不同代码的测试
```python
weights = './weights/yolo_checkpoint/下载的权重.pt'  
yolo = False
```
设置分类信息
```python
classify_id = ['KZ']
video_path = 'QQ视频20231122174123.mp4'
```
注意：当使用非ultralytics 模型时，需要对detector进行修改 93行
```python
detector = Detector(weights, classify_id, yaml, yolo=yolo)
```

## 5. 结果
通过设置output保存输出影像的地址
```python
output = 'result.mp4' # 保存图像
```
## 6. 训练
训练模型见：github  

## 7. 问题
常见问题可以咨询在github，有特殊问题可以咨询hpu2xld@gmail.com

## 8. Citation
if this work is helpful to you,Please note the source.
