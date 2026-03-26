# with open(r"output.txt", 'r') as file:
#     lines = file.readlines()
#
# # 用字典存储序列，按标签分组
# sequences = {}
# for line in lines:
#     parts = line.split()
#     if len(parts) == 2:
#         label = parts[0]
#         sequence = parts[1]
#         if label not in sequences:
#             sequences[label] = []
#         sequences[label].append(sequence.strip())
#
# # 将字典按键排序
# sorted_labels = sorted(sequences.keys(), key=int)
#
# # 创建并写入sequence.txt文件
# with open(r"original.txt", 'w') as fasta_file:
#     for label in sorted_labels:
#         #fasta_file.write(f">{label}\n")
#         for seq in sequences[label]:
#             fasta_file.write(f"{seq}\n")
#         fasta_file.write("===============================\n")



# with open(r"output.txt", 'r') as file:
#     lines = file.readlines()
#
# # 用字典存储序列，按标签分组
# sequences = {}
# for line in lines:
#     parts = line.split()
#     if len(parts) == 2:
#         label = parts[0]
#         sequence = parts[1]
#         if label not in sequences:
#             sequences[label] = []
#         sequences[label].append(sequence.strip())
#
# # 将字典按键排序
# sorted_labels = sorted(sequences.keys(), key=int)
# # sorted_labels = sorted(sequences.keys(), key=lambda x: int(x.replace('seq', '')))
#
#
# # 输出字典中序列的个数和簇的数量
# total_sequences = sum(len(sequences[label]) for label in sorted_labels)  # 总序列数
# total_clusters = len(sorted_labels)  # 簇的数量
#
# print(f"字典中存储的序列个数: {total_sequences}")
# print(f"形成的簇的数量: {total_clusters}")
#
# # 创建并写入 original.txt 文件
# with open(r"original.txt", 'w') as fasta_file:
#     for label in sorted_labels:
#         for seq in sequences[label]:
#             fasta_file.write(f"{seq}\n")
#         fasta_file.write("===============================\n")  # 每个簇后添加分隔符
#
# print("处理完成，结果保存在 original.txt 文件中。")

# def remove_prefix_until_AGTG(input_file, output_file):
#     with open(input_file, 'r') as file:
#         lines = file.read().splitlines()
#
#     modified_sequences = []
#
#     for line in lines:
#         if line == '===============================':
#             # 保留分隔符
#             modified_sequences.append(line)
#             continue
#
#         # 找到第一个 AGTG 的位置
#         index = line.find('AGTGCAACAAGTCAATCCGT')
#         if index != -1:
#             # 截取从第一个 AGTG 开始的部分
#             modified_sequences.append(line[index:])
#         else:
#             modified_sequences.append(line)  # 如果没有 AGTG，保留原序列
#
#     # 处理结束后，确保输出文件的格式
#     with open(output_file, 'w') as file:
#         file.write('\n'.join(modified_sequences) + '\n')
#
#     print(f"处理完成，结果保存在 {output_file} 文件中。")
#
#
# # 使用示例
# remove_prefix_until_AGTG('original.txt',
#                          'modified_original.txt')

def remove_prefix_suffix_AGTG_GCCG(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.read().splitlines()

    # 统计处理前的簇数量（每个分隔符代表一个簇）
    num_clusters_before = lines.count('===============================')

    modified_sequences = []

    for line in lines:
        if line == '===============================':
            modified_sequences.append(line)
            continue

        start_index = line.find('AGTG')
        end_index = line.rfind('GCCG')

        if start_index != -1 and end_index != -1 and start_index < end_index:
            modified_sequences.append(line[start_index:end_index + 4])
        else:
            continue  # 删除整条序列

    # 统计处理后的簇数量（分隔符数量）
    num_clusters_after = modified_sequences.count('===============================')

    with open(output_file, 'w') as file:
        file.write('\n'.join(modified_sequences) + '\n')

    print(f"✅ 处理完成，结果保存在 {output_file} 文件中。")
    print(f"🔢 处理前簇数量：{num_clusters_before}")
    print(f"🔢 处理后簇数量：{num_clusters_after}")
    if num_clusters_after < num_clusters_before:
        print("⚠️ 注意：部分簇的分隔符可能在输入中遗漏或格式异常。")

# 使用示例
remove_prefix_suffix_AGTG_GCCG('original.txt',
                         'modified_original.txt')

