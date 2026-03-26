
import random

def process_sequences(input_file, reference_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.read().splitlines()

    with open(reference_file, 'r') as file:
        references = file.read().strip().splitlines()

    clusters = []
    current_cluster = []
    references_to_keep = []



    # 按簇分组
    for line in lines:
        if line == '===============================':
            clusters.append(current_cluster)
            current_cluster = []
            # if current_cluster:
            #     clusters.append(current_cluster)
            #     current_cluster = []
        else:
            current_cluster.append(line)

    if current_cluster:  # 添加最后一个簇
        clusters.append(current_cluster)

    print(f"读取到 {len(clusters)} 个簇，对应 {len(references)} 条参考序列。")

    if len(clusters) != len(references):
        print("❌ 错误：簇的数量与参考序列数量不一致！")
    else:
        print("✅ 数量一致，开始检查每个簇是否合法...")

    random.seed(0)  # 固定随机种子以确保结果可重复

    with open(output_file, 'w') as file:
        for i, cluster in enumerate(clusters):
            valid_sequences = [seq for seq in cluster if 145 <= len(seq) <= 155 and 'N' not in seq]
            if valid_sequences:
                references_to_keep.append(references[i])  # 记录对应的参考序列
                # file.write('\n'.join(valid_sequences) + '\n')
                if len(valid_sequences) > 30:
                    group_size = random.randint(1, 30)
                    selected_sequences = random.sample(valid_sequences, min(group_size, len(valid_sequences)))
                    file.write('\n'.join(selected_sequences) + '\n')
                else:
                    file.write('\n'.join(valid_sequences) + '\n')
                file.write('===============================\n')

    # 更新参考序列
    with open('reference.txt', 'w') as file:
        file.write('\n'.join(references_to_keep) + '\n')

    print(f"处理完成，结果保存在 {output_file} 文件中。")

process_sequences('modified_original.txt', 'ref.txt', 'reads.txt')


# import random
#
# def process_sequences(input_file, reference_file, output_file):
#     prefix = "ACACTCTTTCCCTACACGACGCTCTTCCGATCT"
#     suffix = "AGATCGGAAGAGCGGTTCAGCAGGAATGCCGAG"
#
#     with open(input_file, 'r') as file:
#         lines = file.read().splitlines()
#
#     with open(reference_file, 'r') as file:
#         references = file.read().strip().splitlines()
#
#     clusters = []
#     current_cluster = []
#     references_to_keep = []
#
#     # 按簇分组
#     for line in lines:
#         if line == '===============================':
#             if current_cluster:
#                 clusters.append(current_cluster)
#                 current_cluster = []
#         else:
#             current_cluster.append(line)
#
#     if current_cluster:  # 添加最后一个簇
#         clusters.append(current_cluster)
#
#     random.seed(0)  # 固定随机种子以确保结果可重复
#
#     with open(output_file, 'w') as file:
#         for i, cluster in enumerate(clusters):
#             valid_sequences = [seq for seq in cluster if 112 <= len(seq) <= 122 and 'N' not in seq]
#             if valid_sequences:
#                 references_to_keep.append(references[i])  # 记录对应的参考序列
#                 # 为有效序列添加前缀和后缀
#                 valid_sequences = [f"{prefix}{seq}{suffix}" for seq in valid_sequences]
#                 if len(valid_sequences) > 30:
#                     group_size = random.randint(1, 30)
#                     selected_sequences = random.sample(valid_sequences, min(group_size, len(valid_sequences)))
#                     file.write('\n'.join(selected_sequences) + '\n')
#                 else:
#                     file.write('\n'.join(valid_sequences) + '\n')
#                 file.write('===============================\n')
#
#     # 更新参考序列
#     with open(reference_file, 'w') as file:
#         file.write('\n'.join(references_to_keep) + '\n')
#
#     print(f"处理完成，结果保存在 {output_file} 文件中。")
#
# # 调用函数
# process_sequences('/amax/linhaiyan_data/PE_AYB/original.txt',
#                   '/amax/linhaiyan_data/PE_AYB/reference.txt',
#                   '/amax/linhaiyan_data/PE_AYB/reads.txt')

# import random
#
# def process_sequences(input_file, reference_file, output_file):
#     with open(input_file, 'r') as file:
#         lines = file.read().splitlines()
#
#     with open(reference_file, 'r') as file:
#         references = file.read().strip().splitlines()
#
#     clusters = []
#     current_cluster = []
#     references_to_keep = []
#
#     # 按簇分组
#     for line in lines:
#         if line == '===============================':
#             if current_cluster:
#                 clusters.append(current_cluster)
#                 current_cluster = []
#         else:
#             current_cluster.append(line)
#
#     if current_cluster:  # 添加最后一个簇
#         clusters.append(current_cluster)
#
#     print(f"读取到 {len(clusters)} 个簇，对应 {len(references)} 条参考序列。")
#
#     if len(clusters) != len(references):
#         print("❌ 错误：簇的数量与参考序列数量不一致！")
#     else:
#         print("✅ 数量一致，开始处理每个簇...")
#
#     random.seed(0)  # 固定随机种子以确保结果可重复
#
#     with open(output_file, 'w') as file:
#         for i, cluster in enumerate(clusters):
#             processed_sequences = []
#
#             # 处理每条序列：截取从指定片段开始
#             for seq in cluster:
#                 idx = seq.find('AGTG')
#                 # idx = seq.find('AGTGCAACAAGTCAATCCG')
#                 if idx != -1:
#                     clipped = seq[idx:]
#                     if 145 <= len(clipped) <= 155 and 'N' not in clipped:
#                         processed_sequences.append(clipped)
#
#             # 如果合法序列不少于5，则保留
#             if len(processed_sequences) >= 5:
#                 references_to_keep.append(references[i])
#                 if len(processed_sequences) > 30:
#                     group_size = random.randint(5, 30)
#                     selected = random.sample(processed_sequences, group_size)
#                 else:
#                     selected = processed_sequences
#                 file.write('\n'.join(selected) + '\n')
#                 file.write('===============================\n')
#             # else:
#             #     print(f"❌ 簇 {i} 被删除（合法序列数 = {len(processed_sequences)}）")
#
#     # 更新参考序列
#     with open(reference_file, 'w') as file:
#         file.write('\n'.join(references_to_keep) + '\n')
#
#     print(f"✅ 处理完成，结果保存在 {output_file} 文件中。")
#
# # 调用函数
# process_sequences('original.txt', 'reference.txt', 'reads.txt')




