'***转换xml标注文件为txt格式，无法直接运行***'
import copy
from lxml.etree import Element, SubElement, tostring, ElementTree

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["1"]  # 类别 设置为你标注时的类别


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id, path):

    in_path = path + '\\labels_xml'
    out_path = path + '\\labels'
    # 确保文件夹存在
    if not os.path.exists(in_path):
        print("文件夹不存在！")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    in_file = open(in_path + '\\%s.xml' % (image_id), encoding='UTF-8')

    out_file = open(out_path + '/%s.txt' % (image_id), 'w')  # 生成txt格式文件, 保存在yolov7训练所需的数据集路径中

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        print(cls)
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

xml_path = 'C:\\Users\\xld77\\OneDrive\\桌面\\chopsticks\\labels_xml'  # xml_path应该是上述步骤OBS桶文件夹C中的所有文件，记得拷贝过来
path = 'C:\\Users\\xld77\\OneDrive\\桌面\\chopsticks'
img_xmls = os.listdir(xml_path)
for img_xml in img_xmls:
    label_name = img_xml.split('.')[0]
    if  img_xml.split('.')[1] == 'xml':
        convert_annotation(label_name, path)