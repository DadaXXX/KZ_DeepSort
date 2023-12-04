import os
import random

def read_filenames(folder_path):
    filenames = []

    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print("文件夹不存在！")
        return filenames

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 确保路径为文件而不是文件夹
        if os.path.isfile(file_path):
            filenames.append(file_path)

    return filenames

# 指定要读取文件名的文件夹路径
folder_path = "C:\\Users\\xld77\\OneDrive\\桌面\\data\\chopsticks\\images"

# 调用函数读取文件名
filenames = read_filenames(folder_path)
random.seed(0)
random.shuffle(filenames)

# 分割数据集一部分用于测试一部分用于训练
trainnums = 0.9
valnums = 0.1
train = int(trainnums * len(filenames))
val = len(filenames) - train

folder_path = "C:\\Users\\xld77\\OneDrive\\桌面\\data\\chopsticks"
# 写入文件
output_train = folder_path + "\\train.txt"
output_val = folder_path + "\\val.txt"
output_test = folder_path + "\\test.txt"

i=0
with open(output_train, "w") as f:

    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")
    for x in range(train):
        f.write(filenames[x] + "\n")


print("文件名已保存到", output_train)
output_file = output_val
with open(output_file, "w") as f:
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")
    for x in range(val):
        x = x + train
        f.write(filenames[x] + "\n")



print("文件名已保存到", output_file)