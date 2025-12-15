import random
import os


def mutate_sequence(seq, sub_rate=0.02, del_rate=0.02, ins_rate=0.02):
    """
    模拟 DNA 测序错误：替换(Substitution), 删除(Deletion), 插入(Insertion)
    这是为了模仿 I18-S3 那种高噪声环境
    """
    bases = ['A', 'C', 'G', 'T']
    new_seq = []

    for base in seq:
        r = random.random()

        # 1. Deletion (删除)
        if r < del_rate:
            continue

        # 2. Insertion (插入)
        if r < del_rate + ins_rate:
            new_seq.append(random.choice(bases))
            new_seq.append(base)  # 插入一个随机碱基，并保留原碱基
            continue

        # 3. Substitution (替换)
        if r < del_rate + ins_rate + sub_rate:
            # 替换成不同于当前的碱基
            candidates = [b for b in bases if b != base]
            new_seq.append(random.choice(candidates))
        else:
            # No Error
            new_seq.append(base)

    return "".join(new_seq)


def generate_simulation_data(
        num_clusters=100,  # 模拟 100 个原本的序列 (Reference)
        reads_per_cluster=20,  # 每个序列生成 20 个带噪副本 (Reads)
        seq_length=152,  # 【注意】序列长度要和你Clover脚本里的 -L 参数一致！
        output_file="raw_reads.txt",
        truth_file="ground_truth_map.txt"  # 新增：保存真值，方便后续验证
):
    bases = ['A', 'C', 'G', 'T']

    # 1. 生成 Ground Truth (原本的完美序列)
    ground_truths = []
    for _ in range(num_clusters):
        seq = "".join(random.choice(bases) for _ in range(seq_length))
        ground_truths.append(seq)

    print(f"生成的 Ground Truth 数量: {len(ground_truths)}")

    # 2. 生成带噪 Reads
    # 格式: ID, Sequence
    all_reads = []

    # 我们用一个字典记录真实归属，方便你后面验证 (训练时不用)
    # truth_map 结构: {read_id: (cluster_index, original_sequence)}
    truth_map_data = []

    read_counter = 0
    for cluster_idx, ref_seq in enumerate(ground_truths):
        for _ in range(reads_per_cluster):
            read_counter += 1
            read_id = str(read_counter)

            # 施加噪声 (模拟 I18-S3 的高误差环境)
            # 这里设置较高的 indel 率，挑战 Clover
            noisy_seq = mutate_sequence(ref_seq, sub_rate=0.02, del_rate=0.03, ins_rate=0.03)

            all_reads.append((read_id, noisy_seq))
            # 记录真值：Read ID, 所属簇ID, 真实序列
            truth_map_data.append(f"{read_id}\t{cluster_idx}\t{ref_seq}")

    # 打乱顺序，模拟真实测序仪输出
    random.shuffle(all_reads)

    # 3. 写入输入文件 (raw_reads.txt)
    with open(output_file, 'w') as f:
        for rid, seq in all_reads:
            # 【关键修改】这里用 \t 分隔，Clover 就能完美识别了
            f.write(f"{rid}\t{seq}\n")

    # 4. 写入真值文件 (ground_truth_map.txt)
    with open(truth_file, 'w') as f:
        f.write("Read_ID\tTrue_Cluster_ID\tTrue_Ref_Seq\n")
        for line in truth_map_data:
            f.write(line + "\n")

    print(f"✅ 模拟数据集生成完毕: {output_file} (已使用 Tab 分隔)")
    print(f"✅ 真值记录文件生成完毕: {truth_file} (请保留好，后续算准确率用)")
    print(f"总 Reads 数: {len(all_reads)}")
    return ground_truths


# ============================
# 运行生成
# ============================
if __name__ == "__main__":
    # 生成模拟数据
    generate_simulation_data()

    print("\n下一步提示：")
    print("1. 请确保你的 run_clover.sh 脚本中 READ_LENGTH=150 (和上面代码一致)")