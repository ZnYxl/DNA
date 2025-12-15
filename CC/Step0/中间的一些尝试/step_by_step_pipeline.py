import random
import collections
import os

# ==========================================
# 第一步：模拟原始数据 (我们要造假数据啦！)
# ==========================================
print(">>> 正在生成模拟的DNA原始数据...")

# 1. 定义4个真实的参考序列 (Ground Truth) -> 这就是未来的 reference.txt 的来源
# 假设我们有4个簇 (Cluster)
original_references = {
    'Ref_1': 'AGTGCAACAAGTCAATCCGTGTCGCTATGCATGTGTCTACGTATGCACTACGTGATGTCTGCATCATGCAGTACGCGCGAGCACGCGCTATGCACAGTCGCTCGCTATCTATATACTATCGTCGTGACAGAATTGAATGCTTGCTTGCCG',
    'Ref_2': 'TTGGTCTAGGCGTTTTCTGTACTTGAGAGCTTTTGCTAGCCTCCGAACTACCGCAGTCTGACTCTGCGATTGGAGCCTCGAGACCGGAACAACCGACCCCCCTACTCGATGAATCACATATTTAAGACGGCCAGACTCCCTATGGTTGGT',
    'Ref_3': 'GCGTAACCGTATTGATGATGGTAGCGCACCTAAAGCATACTTGACGTAGTGCTATCCCTTCGCCTGTCCTCCACGGAAATGGCGGTTATTCAGAGCACGTTCCATTAGACATAACAATCCAGAGTGTGTAAGATATATGCGAGAGATTTC',
    'Ref_4': 'CATCATACAGCAGCATGTGTGATGTACAGCAGTATGTGATGAGTAGATGAGCGCTCACGACTCAGTAGTGACTGATAGTGTCTACAGTAGCGTATCTAGTAGTCGCTAGAATTGAATGCTTGCTTGCCG'
}

# 2. 生成带噪音的 Reads (模拟测序仪的输出)
# 我们让每个 Reference 生成 3 条 noisy reads
raw_reads_dict = {}  # 格式: { 'read_id': 'DNA_SEQUENCE...' }
true_label_map = {}  # 记录每个 Read 真正属于哪个 Ref (为了验证)

read_counter = 0
for ref_id, seq in original_references.items():
    # 每个 Ref 生成 3 个变体
    for i in range(3):
        read_counter += 1
        read_id = str(read_counter)

        # 模拟一点点错误 (随机把几个碱基换掉)
        noisy_seq = list(seq)
        if len(seq) > 10:
            idx = random.randint(0, len(seq) - 1)
            noisy_seq[idx] = 'N'  # 这里的N代表错误或突变

        raw_reads_dict[read_id] = "".join(noisy_seq)
        true_label_map[read_id] = ref_id

print(f"模拟完成：共生成 {len(raw_reads_dict)} 条 Reads。")
print(f"示例 Read 1: {raw_reads_dict['1'][:20]}...")

# ==========================================
# 第二步：模拟 Clover 的工作 (假装跑了聚类)
# ==========================================
print("\n>>> 正在模拟 Clover 聚类过程...")

# Clover 的输出通常是：这一条 Read (Member) 属于 哪一条 Read (Center)
# 我们手动构造一个完美的聚类结果：
# 假设：
#   Read 1, 2, 3 属于 Cluster A (以 Read 1 为中心)
#   Read 4, 5, 6 属于 Cluster B (以 Read 4 为中心)
#   ...以此类推

clover_output_list = [
    # (成员ID, 中心ID)
    ('1', '1'), ('2', '1'), ('3', '1'),  # 簇 1
    ('4', '4'), ('5', '4'), ('6', '4'),  # 簇 2
    ('7', '7'), ('8', '7'), ('9', '7'),  # 簇 3
    ('10', '10'), ('11', '10'), ('12', '10')  # 簇 4
]

print("Clover 输出模拟完毕。结构如下:")
print(clover_output_list)

# ==========================================
# 第三步：连接 FedDNA (你要的脚本就在这！)
# ==========================================
print("\n>>> 正在转换为 FedDNA 格式 (read.txt 和 reference.txt)...")


def convert_to_feddna(clover_output, raw_reads, true_labels, output_folder="./"):
    # 1. 整理聚类
    clusters = collections.defaultdict(list)
    for member_id, center_id in clover_output:
        clusters[center_id].append(member_id)

    output_reads_path = os.path.join(output_folder, "read.txt")
    output_ref_path = os.path.join(output_folder, "reference.txt")

    with open(output_reads_path, 'w') as f_read, open(output_ref_path, 'w') as f_ref:

        cluster_count = 0
        for center_id, member_ids in clusters.items():
            cluster_count += 1

            # --- 写 read.txt ---
            for mid in member_ids:
                seq = raw_reads.get(mid)
                if seq:
                    f_read.write(seq + '\n')

            # FedDNA 的关键分隔符
            f_read.write("===============================\n")

            # --- 写 reference.txt ---
            # 逻辑：找到这个簇对应的真实 Reference
            # 我们用 center_id 去查 true_label_map，看看它属于哪个 Ref
            # 然后去 original_references 里拿真正的序列

            ref_key = true_labels.get(center_id)  # 比如 'Ref_1'
            real_ref_seq = original_references.get(ref_key)

            if real_ref_seq:
                f_ref.write(real_ref_seq + '\n')
            else:
                f_ref.write("ERROR_NO_REF_FOUND\n")

    print(f"转换成功！生成了 {cluster_count} 个簇。")
    print(f"文件已保存至: {os.path.abspath(output_folder)}")


# 执行转换
convert_to_feddna(clover_output_list, raw_reads_dict, true_label_map)

print("\n>>> 恭喜！现在你可以打开文件夹查看 read.txt 和 reference.txt 了。")