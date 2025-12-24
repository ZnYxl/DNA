import os
import random
import ast
import collections
import numpy as np
from random import seed, shuffle, randint, choice
import multiprocessing  # ✅ 新增多进程库
from functools import partial

# ==============================================================================
# Part 1: 师兄提供的模拟代码 (完全保留，原封不动)
# ==============================================================================

rd = random.Random() 

def channel_model_unit(code, pr_dict):
    # 信道模拟函数
    del_num = 0
    ins_num = 0
    sub_num = 0
    pt_num = 0
    unit_list = ["A","T","G","C"]
    af_code = ""
    if rd.random() <= pr_dict["column"]:
        return ""
    else:
        for i in range(len(code)):
            ins_times = 0
            while ins_times < 1:  
                if rd.random() <= pr_dict["pi"]:
                    af_code += random.choice(unit_list)
                    ins_num = ins_num + 1
                else:
                    break
            if rd.random() <= pr_dict["pd"]:
                del_num += 1
                continue
            else:
                pt_num += 1
                if rd.random() <= pr_dict["ps"]:
                    target = choice(list(filter(lambda base: base != code[i], ["A", "C", "G", "T"])))
                    sub_num += 1
                    af_code+=target
                else:
                    af_code+=code[i]
    return af_code

def channel_simulation(dna_reads_list, depth, random_sample=False, pr_dict={"column":0,"pi":0,"pd":0,"ps":0}):
    channel_reads_list = []
    dna_nums = len(dna_reads_list)
    seq_nums = dna_nums*depth

    if random_sample == True:
        for _ in range(seq_nums):
            index = random.randint(0,dna_nums-1)
            now_read = dna_reads_list[index]
            channel_reads_list.append(channel_model_unit(now_read,pr_dict))
        return channel_reads_list
    else:
        for read in dna_reads_list:
            for _ in range(depth):
                channel_reads_list.append(channel_model_unit(read,pr_dict))
        shuffle(channel_reads_list)  
        return channel_reads_list

# ==============================================================================
# Part 2: 辅助函数 
# ==============================================================================

def generate_diverse_references(num_clusters, seq_len, min_distance=0.3):
    """生成具有足够区分度的参考序列"""
    bases = ['A', 'C', 'G', 'T']
    references = []
    # 为了速度，大数量时简化距离检查，主要依赖随机性
    # 10000条随机序列碰撞概率极低
    for i in range(num_clusters):
        candidate = "".join(random.choice(bases) for _ in range(seq_len))
        references.append(candidate)
    return references

# ==============================================================================
# Part 3: 并行化处理单元 (新增)
# ==============================================================================

def process_single_cluster(args):
    """
    单个簇的处理函数，用于多进程调用
    """
    cluster_idx, ref_seq, reads_per_cluster, pr_dict = args
    
    # 模拟簇大小波动
    low = int(reads_per_cluster * 0.6)
    high = int(reads_per_cluster * 1.4)
    actual_depth = random.randint(low, high)
    
    # 调用师兄代码
    simulated_reads = channel_simulation(
        dna_reads_list=[ref_seq], 
        depth=actual_depth, 
        random_sample=False, 
        pr_dict=pr_dict
    )
    
    results = []
    for r_seq in simulated_reads:
        if not r_seq: continue
        # 返回时不带 ID，由主进程统一分配 ID，避免冲突
        # 格式: (Seq, ClusterID, RefSeq, Quality)
        results.append((r_seq, cluster_idx, ref_seq, "simulated"))
        
    return results

# ==============================================================================
# Part 4: 数据生成主控 (多进程版)
# ==============================================================================

def generate_data(output_dir, num_clusters=100, reads_per_cluster=50, seq_len=150, reference_type="diverse"):
    raw_path = os.path.join(output_dir, "raw_reads.txt")
    read_gt_path = os.path.join(output_dir, "ground_truth_reads.txt")
    cluster_gt_path = os.path.join(output_dir, "ground_truth_clusters.txt")
    
    pr_dict = {"column": 0.0001, "pi": 0.0005, "pd": 0.0005, "ps": 0.008}
    print(f"🔧 [Generator] 多进程加速启动。参数: {pr_dict}")

    # 1. 生成参考序列
    print("   ... 生成 Payload 参考序列")
    ground_truths = generate_diverse_references(num_clusters, seq_len)
    
    with open(cluster_gt_path, 'w') as f:
        f.write("Cluster_ID\tRef_Seq\n")
        for cid, seq in enumerate(ground_truths):
            f.write(f"{cid}\t{seq}\n")

    # 2. 准备并行任务
    print(f"   ... 正在启动多进程池 (Clusters: {num_clusters})")
    
    # 准备参数列表
    tasks = []
    for cluster_idx, ref_seq in enumerate(ground_truths):
        tasks.append((cluster_idx, ref_seq, reads_per_cluster, pr_dict))
    
    # 获取CPU核心数 (保留2个核心给系统)
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    all_reads_data = []
    counter = 0
    
    # ✅ 开启并行处理
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 稍微快一点，且能显示进度
        for result_batch in pool.imap_unordered(process_single_cluster, tasks, chunksize=100):
            for item in result_batch:
                counter += 1
                read_id = str(counter)
                # item是 (Seq, ClusterID, RefSeq, Quality)
                # 加上 read_id
                all_reads_data.append((read_id,) + item)
            
            if len(all_reads_data) % 100000 == 0:
                print(f"      已生成 {len(all_reads_data)} 条 reads...")

    print("   ... 正在打乱数据 (Shuffle)")
    random.shuffle(all_reads_data)

    # 3. 写入文件
    print(f"   ... 写入硬盘 ({len(all_reads_data)} 条)")
    with open(raw_path, 'w') as f:
        for item in all_reads_data:
            f.write(f"{item[0]}\t{item[1]}\n")
            
    with open(read_gt_path, 'w') as f:
        f.write("Read_ID\tCluster_ID\tRef_Seq\tQuality\n")
        for item in all_reads_data:
            f.write(f"{item[0]}\t{item[2]}\t{item[3]}\t{item[4]}\n")
            
    return raw_path, read_gt_path, cluster_gt_path

# ==============================================================================
# Part 5: 格式转换 (Clover -> FedDNA) - 保持不变
# ==============================================================================

def load_raw_reads(file_path):
    d = {}
    with open(file_path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: d[p[0]] = p[1]
    return d

def clover_to_feddna(clover_out_path, raw_reads_path, output_dir):
    # ... (保持原代码不变) ...
    raw_reads = load_raw_reads(raw_reads_path)
    clusters = collections.defaultdict(list)
    try:
        with open(clover_out_path, 'r') as f:
            content = f.read().strip()
            if content.startswith("[") and content.endswith("]"):
                pairs = ast.literal_eval(content)
                for item in pairs:
                    if str(item[1]) not in ['-1', -1]:
                        clusters[str(item[1])].append(str(item[0]))
            else:
                f.seek(0)
                for line in f:
                    p = line.replace(',', ' ').split()
                    if len(p) >= 2 and p[1] not in ['-1', '-1']:
                        clusters[p[1]].append(p[0])
    except Exception as e:
        print(f"❌ 解析 Clover 输出失败: {e}")
        return 0, ""

    out_read = os.path.join(output_dir, "read.txt")
    out_ref = os.path.join(output_dir, "ref.txt")
    
    valid_count = 0
    with open(out_read, 'w') as fr, open(out_ref, 'w') as ff:
        for cid, mems in clusters.items():
            if cid in raw_reads:
                ff.write(raw_reads[cid] + "\n") 
                fr.write(raw_reads[cid] + "\n") 
                for m in mems:
                    if m in raw_reads:
                        fr.write(raw_reads[m] + "\n")
                fr.write("===============================\n")
                valid_count += 1
    return valid_count, out_read