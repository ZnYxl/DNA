# # Read the input FASTA file
# with open(r'/amax/linhaiyan_data/P10_5_BDDP210000009/output.fasta', 'r') as fasta_file:
#     lines = fasta_file.readlines()
#
# # Process the lines to remove headers and store sequences
# sequences = []
# for line in lines:
#     if not line.startswith('>'):
#         sequences.append(line.strip())
#
# # Write the sequences to the output file
# with open('/amax/linhaiyan_data/P10_5_BDDP210000009/reference.txt', 'w') as output_file:
#     for sequence in sequences:
#         output_file.write(sequence + '\n')


# import os
#
# input_file = r"/mnt/st_data/linhaiyan/data-nbt17/id20.refs.txt"
# output_file = r"design.txt"
#
#
# print("开始读取文件...")
#
# try:
#     with open(input_file, 'r') as f:
#         lines = f.readlines()
#
#
#     # 处理每一行，删除特定前后缀
#     cleaned_lines = [line.strip().replace("5'-", "").replace("-3'", "") for line in lines]
#
#
#     # 写入到输出文件
#     with open(output_file, 'w') as f:
#         for line in cleaned_lines:
#             f.write(line + '\n')
#     print(f"已成功保存到: {output_file}")
#
# except Exception as e:
#     print("发生错误:", e)


# 读取 output.txt，提取序列的数字
with open('output.txt', 'r') as f:
    output_lines = f.readlines()
    output_numbers = {line.split()[0] for line in output_lines}  # 提取数字

# 输出不同数字的个数
num_unique_numbers = len(output_numbers)
print(f"不同数字的个数: {num_unique_numbers}")

# 读取 design.txt，按顺序排序
with open('design.txt', 'r') as f:
    design_lines = f.readlines()

# 创建一个新的列表，保留符合条件的序列
filtered_sequences = []
for i, seq in enumerate(design_lines):
    seq = seq.strip()  # 去掉前后空格
    # 使用 i + 1 作为排序的数字（从 1 开始）
    if str(i + 1) in output_numbers:
        filtered_sequences.append(seq)

# 保存到 reference.txt
with open('ref.txt', 'w') as f:
    for seq in filtered_sequences:
        f.write(seq + '\n')

# 输出 reference.txt 中的序列数
num_sequences_in_reference = len(filtered_sequences)
print(f"ref.txt 中的序列数: {num_sequences_in_reference}")













