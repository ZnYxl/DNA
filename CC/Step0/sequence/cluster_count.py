# def count_sequences_in_clusters(input_file):
#     with open(input_file, 'r') as file:
#         content = file.read().splitlines()
#
#     cluster_counts = []
#     current_count = 0
#
#     for line in content:
#         if line == '===============================':
#             if current_count > 0:  # 只在当前计数大于0时添加
#                 cluster_counts.append(current_count)
#                 current_count = 0  # 重置计数
#         else:
#             current_count += 1  # 统计序列
#
#     # 添加最后一个簇的计数（如果有）
#     if current_count > 0:
#         cluster_counts.append(current_count)
#
#     # 去重并排序
#     unique_counts = sorted(set(cluster_counts))
#
#     print("每个簇之间的序列数量（去重后）：")
#     for count in unique_counts:
#         print(count)
#
# # 使用示例
# count_sequences_in_clusters(r'/amax/linhaiyan_data/P10_5_BDDP210000009/original.txt')


def count_clusters(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 统计簇的数量
    cluster_count = 0
    for line in lines:
        if line.strip() == "===============================":
            cluster_count += 1



    return cluster_count

# 使用示例
file_path = "reads.txt"
num_clusters = count_clusters(file_path)
print(f"簇的数量: {num_clusters}")

