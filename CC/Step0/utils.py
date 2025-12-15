# utils.py
import random
import os
import ast
import collections


# ==================== 模块1: 数据生成 ====================
def mutate_sequence(seq, sub_rate, del_rate, ins_rate):
    """模拟DNA错误"""
    bases = ['A', 'C', 'G', 'T']
    new_seq = []
    for base in seq:
        r = random.random()
        if r < del_rate: continue
        if r < del_rate + ins_rate:
            new_seq.append(random.choice(bases))
            new_seq.append(base)
            continue
        if r < del_rate + ins_rate + sub_rate:
            new_seq.append(random.choice([b for b in bases if b != base]))
        else:
            new_seq.append(base)
    return "".join(new_seq)


def generate_data(output_dir, num_clusters=100, reads_per_cluster=20, seq_len=150):
    """生成模拟数据并保存到指定目录"""
    raw_path = os.path.join(output_dir, "raw_reads.txt")
    gt_path = os.path.join(output_dir, "ground_truth.txt")

    bases = ['A', 'C', 'G', 'T']
    ground_truths = ["".join(random.choice(bases) for _ in range(seq_len)) for _ in range(num_clusters)]

    all_reads = []
    truth_data = []
    counter = 0

    for idx, ref in enumerate(ground_truths):
        for _ in range(reads_per_cluster):
            counter += 1
            # 这里可以调整错误率
            noisy = mutate_sequence(ref, sub_rate=0.02, del_rate=0.03, ins_rate=0.03)
            rid = str(counter)
            all_reads.append((rid, noisy))
            truth_data.append(f"{rid}\t{idx}\t{ref}")

    random.shuffle(all_reads)

    with open(raw_path, 'w') as f:
        for rid, seq in all_reads:
            f.write(f"{rid}\t{seq}\n")  # 使用 Tab 分隔

    with open(gt_path, 'w') as f:
        f.write("Read_ID\tCluster_ID\tRef_Seq\n")
        for line in truth_data:
            f.write(line + "\n")

    return raw_path, gt_path


# ==================== 模块2: 格式转换 ====================
def load_raw_reads(file_path):
    d = {}
    with open(file_path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: d[p[0]] = p[1]
    return d


def clover_to_feddna(clover_out_path, raw_reads_path, output_dir):
    """将Clover结果转换为FedDNA格式"""
    raw_reads = load_raw_reads(raw_reads_path)
    clusters = collections.defaultdict(list)

    # 智能解析 Clover 输出
    with open(clover_out_path, 'r') as f:
        content = f.read().strip()
        if content.startswith("[") and content.endswith("]"):
            # 列表格式解析
            try:
                pairs = ast.literal_eval(content)
                for item in pairs:
                    if str(item[1]) not in ['-1', -1]:
                        clusters[str(item[1])].append(str(item[0]))
            except:
                print("解析列表格式失败")
        else:
            # 逐行格式解析
            f.seek(0)
            for line in f:
                p = line.replace(',', ' ').split()
                if len(p) >= 2 and p[1] not in ['-1', '-1']:
                    clusters[p[1]].append(p[0])

    # 写入结果
    out_read = os.path.join(output_dir, "read.txt")
    out_ref = os.path.join(output_dir, "reference.txt")

    valid_count = 0
    with open(out_read, 'w') as fr, open(out_ref, 'w') as ff:
        for cid, mems in clusters.items():
            if cid in raw_reads:
                ff.write(raw_reads[cid] + "\n")  # 伪Reference
                for m in mems:
                    if m in raw_reads:
                        fr.write(raw_reads[m] + "\n")
                fr.write("===============================\n")
                valid_count += 1

    return valid_count, out_read