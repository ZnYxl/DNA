import os
import random

# 设置数据路径
data_path = '/mnt/st_data/linhaiyan/sequence'
save_path = '/mnt/st_data/linhaiyan/sequence'

# 读取文件
with open(os.path.join(data_path, 'reads.txt'), 'r') as f:
    content = f.read().strip()
    if content.endswith('==============================='):
        content = content[:-len('===============================')]
    reads = content.split('===============================\n')

with open(os.path.join(data_path, 'reference.txt'), 'r') as f:
    references = f.read().strip().splitlines()

print(len(reads), len(references))

# 确保数量匹配
assert len(reads) == len(references)

# 创建数据集分配
data = list(zip(reads, references))
# random.shuffle(data)

train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.2)

# 划分数据集
train_data = data[:1000]
val_data = data[1000:2000]
test_data = data[2000:12000]

# 保存文件
os.makedirs(os.path.join(save_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'val'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)

# 写入训练集
with open(os.path.join(save_path, 'train', 'reads.txt'), 'w') as f:
    for read, _ in train_data:
        f.write(read + '===============================\n')

with open(os.path.join(save_path, 'train', 'reference.txt'), 'w') as f:
    for _, reference in train_data:
        f.write(reference + '\n')

# 写入验证集
with open(os.path.join(save_path, 'val', 'reads.txt'), 'w') as f:
    for read, _ in val_data:
        f.write(read + '===============================\n')

with open(os.path.join(save_path, 'val', 'reference.txt'), 'w') as f:
    for _, reference in val_data:
        f.write(reference + '\n')

# 写入测试集
with open(os.path.join(save_path, 'test', 'reads.txt'), 'w') as f:
    for read, _ in test_data:
        f.write(read + '===============================\n')

with open(os.path.join(save_path, 'test', 'reference.txt'), 'w') as f:
    for _, reference in test_data:
        f.write(reference + '\n')

