import os

def rename_files(folder_path):
    # 获取文件夹中的文件列表
    files = os.listdir(folder_path)

    i = 77
    for file in files:
        # 构造新的文件名
        new_name = str(i)+ '.png'
        # new_name = file.replace(".xml", "")

        # 构造文件的完整路径
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名文件: {file} -> {new_name}")
        i += 1

# 指定文件夹路径和前缀
folder_path = "C:\\Users\\xld77\\OneDrive\\桌面\\kzimage"


# 调用函数进行文件重命名
rename_files(folder_path)