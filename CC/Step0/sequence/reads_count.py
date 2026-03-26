# 读取文件内容
with open(r'/amax/linhaiyan_data/data-nbt17-master/original.txt', 'r') as file:
    content = file.read()

# 将内容按行分割
lines = content.split('\n')

# 去掉分隔符行
sequences = [line for line in lines if line and line != '===============================']

# 统计每个序列的长度及其条数
length_counts = {}
for seq in sequences:
    length = len(seq)
    if length in length_counts:
        length_counts[length] += 1
    else:
        length_counts[length] = 1

# 按序列长度排序
sorted_length_counts = sorted(length_counts.items())

# 输出总序列条数
total_sequences = len(sequences)
print(f"Total DNA sequences: {total_sequences}")

# 输出每种长度的序列条数
for length, count in sorted_length_counts:
    print(f"Length: {length}, Count: {count}")
