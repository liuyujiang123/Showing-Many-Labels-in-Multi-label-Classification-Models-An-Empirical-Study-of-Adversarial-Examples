import os

# 设置包含.txt文件的目录
directory = '../data/VOC2007/test/VOC2007/ImageSets/Main'

# 遍历目录中的文件
if not os.path.exists(directory):
    os.makedirs(directory)
for filename in os.listdir(directory):
    # 检查文件名是否以'_test'结尾
    if filename.endswith('1.txt'):
        # 替换'test'为'1'
        new_filename = filename.replace('1', '_mlliw_adv')
        # 构造完整的文件路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed "{filename}" to "{new_filename}"')