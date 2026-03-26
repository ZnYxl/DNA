# 读取文件中的序列
with open('/amax/linhaiyan_data/P10_5_BDDP210000009/reads.txt', 'r') as file:
    sequence = file.read().strip()

# 找出序列中包含的碱基
unique_bases = set(sequence)

# 输出序列中包含的碱基
print("序列中包含的碱基有：")
for base in unique_bases:
    print(base)
