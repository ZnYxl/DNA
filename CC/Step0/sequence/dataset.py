import os
import random

# 读取文件
with open('reads.txt', 'r') as f:
    content = f.read().strip()
    if content.endswith('==============================='):
        content = content[:-len('===============================')]
    reads = content.split('===============================\n')

with open('reference.txt', 'r') as f:
    references = f.read().strip().splitlines()

print(len(reads), len(references))

# 确保数量匹配
assert len(reads) == len(references)

# 创建数据集分配
data = list(zip(reads, references))
random.shuffle(data)

train_size = int(len(data) * 0.7)
train_data = data[:1000]
test_data = data[1000:11000]

# 保存文件
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# 写入训练集
with open('train/reads.txt', 'w') as f:
    for read, _ in train_data:
        f.write(read + '===============================\n')

with open('train/reference.txt', 'w') as f:
    for _, reference in train_data:
        f.write(reference + '\n')

# 写入测试集
with open('test/reads.txt', 'w') as f:
    for read, _ in test_data:
        f.write(read + '===============================\n')

with open('test/reference.txt', 'w') as f:
    for _, reference in test_data:
        f.write(reference + '\n')

