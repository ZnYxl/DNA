#剩下的 0.08% 发生了什么？
import os
import argparse
import numpy as np
import multiprocessing
from functools import partial
from collections import defaultdict
import time

# ================= 配置区域 =================
HARDCODED_CONFIG = {
    "gt_file": "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/20251224_155232_Cluster_GT_Test/01_RawData/ground_truth_clusters.txt",
    "read_gt_file": "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/20251224_155232_Cluster_GT_Test/01_RawData/ground_truth_reads.txt",
    # 自动指向去重版文件
    "pred_file": "/mnt/st_data/liangxinyi/code/iterative_results/20251224_155232_Cluster_GT_Test copy/round_3/step2/consensus_sequences_deduplicated.fasta"
}
# ===========================================

def calculate_identity(seq1, seq2):
    if not seq1 or not seq2: return 0.0
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / max(len(seq1), len(seq2))

def load_gt_data(path):
    """加载 GT ID 和 序列"""
    gt_map = {}
    with open(path, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                gt_map[int(parts[0])] = parts[1]
    return gt_map

def load_pred_seqs(path):
    """加载预测序列"""
    seqs = []
    if not os.path.exists(path): return seqs
    with open(path, 'r') as f:
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq: seqs.append("".join(seq))
                seq = []
            else:
                seq.append(line)
        if seq: seqs.append("".join(seq))
    return seqs

def find_recovered_ids_chunk(chunk_preds, gt_map):
    """
    多进程子任务：
    只返回那些被完美找回 (Identity > 0.999) 的 GT ID
    """
    recovered_in_chunk = set()
    gt_items = list(gt_map.items())
    
    for pseq in chunk_preds:
        # 1. 尝试完全匹配 (极速)
        # 注意：这需要反向索引，为了简化多进程逻辑，这里主要靠扫描
        # 如果追求极致速度，可以在主进程做 exact match，这里只做 fuzzy
        
        best_score = 0.0
        best_id = -1
        
        # 优化：一旦找到完美匹配就停止
        for gid, gseq in gt_items:
            # 简单长度过滤，加速比对
            if abs(len(pseq) - len(gseq)) > 5: continue
            
            score = calculate_identity(pseq, gseq)
            if score > 0.999:
                best_score = score
                best_id = gid
                break # 找到了！下一个预测序列
        
        if best_score > 0.999:
            recovered_in_chunk.add(best_id)
            
    return recovered_in_chunk

def analyze_read_coverage(read_gt_path, missing_ids):
    print(f"\n   📊 正在扫描原始 Reads 数据 (分析丢失原因)...")
    counts = defaultdict(int)
    
    with open(read_gt_path, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cid = int(parts[1])
                if cid in missing_ids: # 只记录我们关心的
                    counts[cid] += 1
    
    print(f"\n   🕵️‍♂️ 【丢失簇分析报告】")
    print(f"   GT_ID\tRaw Reads Count\tStatus")
    print("-" * 40)
    
    low_cov_count = 0
    for mid in sorted(list(missing_ids)):
        count = counts[mid]
        flag = "🔴 极低 (物理丢失)" if count < 5 else "⚠️ 需检查"
        if count < 5: low_cov_count += 1
        print(f"   {mid}\t\t{count}\t\t{flag}")
        
    return low_cov_count

def main():
    print(f"🚀 开始高性能丢失簇调查...")
    
    # 1. 加载数据
    gt_map = load_gt_data(HARDCODED_CONFIG['gt_file'])
    all_gt_ids = set(gt_map.keys())
    print(f"   ✅ GT 总数: {len(all_gt_ids)}")
    
    pred_seqs = load_pred_seqs(HARDCODED_CONFIG['pred_file'])
    print(f"   ✅ 预测序列数: {len(pred_seqs)}")
    
    # 2. 多进程比对 (复用 verify_final 的高性能逻辑)
    num_cpus = min(64, multiprocessing.cpu_count())
    print(f"   ⚡ 启动多进程比对 (CPU: {num_cpus})...")
    start_time = time.time()
    
    chunk_size = len(pred_seqs) // num_cpus + 1
    chunks = [pred_seqs[i:i + chunk_size] for i in range(0, len(pred_seqs), chunk_size)]
    
    pool = multiprocessing.Pool(processes=num_cpus)
    func = partial(find_recovered_ids_chunk, gt_map=gt_map)
    
    recovered_ids = set()
    for res_set in pool.map(func, chunks):
        recovered_ids.update(res_set)
        
    pool.close()
    pool.join()
    
    print(f"   ✅ 比对完成! 耗时: {time.time() - start_time:.1f}秒")
    
    # 3. 计算丢失
    missing_ids = all_gt_ids - recovered_ids
    print(f"   ❌ 丢失总数: {len(missing_ids)}")
    
    if len(missing_ids) == 0:
        print("\n🎉 完美！所有簇都找回来了！")
        return

    # 4. 分析原因
    low_cov = analyze_read_coverage(HARDCODED_CONFIG['read_gt_file'], missing_ids)
    
    print("\n" + "="*50)
    print("🔎 最终结论")
    print("="*50)
    print(f"丢失的 {len(missing_ids)} 个簇中，有 {low_cov} 个属于低覆盖度 (<5 reads)。")
    
    if low_cov == len(missing_ids):
        print("\n✅ 结论成立：所有丢失均为物理层面的覆盖度不足导致。算法已达理论极限。")
    else:
        print("\n⚠️ 还有部分高覆盖度簇丢失，请记录 ID 进行个案分析。")

if __name__ == "__main__":
    main()