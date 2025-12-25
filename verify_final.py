import os
import argparse
import numpy as np
from collections import defaultdict
import multiprocessing
from functools import partial
import time
from types import SimpleNamespace # 用于把字典转成对象

# ==========================================
# 📝 【配置区域】修改这里即可！
# ==========================================
CONFIG = {
    # 1. 原始实验数据目录 (包含 01_RawData 的那个文件夹)
    # 根据你刚才的日志，应该是这个路径：
    "exp_dir": "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/20251224_155232_Cluster_GT_Test",
    
    # 2. 迭代结果目录 (包含 round_1, round_2, round_3 的那个文件夹)
    # 根据日志，你的结果存放在这里：
    "result_dir": "./iterative_results/20251224_155232_Cluster_GT_Test",
    
    # 3. 要验证的轮次 (通常验证最后一轮，即第 3 轮)
    "round": 3
}
# ==========================================

def calculate_identity(seq1, seq2):
    """计算两条序列的一致性"""
    if not seq1 or not seq2: return 0.0
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / max(len(seq1), len(seq2))

def load_fasta(path):
    seqs = {}
    if not os.path.exists(path): return seqs
    with open(path, 'r') as f:
        header = None; seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header: seqs[int(header.split('_')[1])] = "".join(seq)
                header = line; seq = []
            else: seq.append(line)
        if header: seqs[int(header.split('_')[1])] = "".join(seq)
    return seqs

def load_gt_clusters(experiment_dir):
    gt_path = os.path.join(experiment_dir, "01_RawData", "ground_truth_clusters.txt")
    seqs = {}
    if not os.path.exists(gt_path): return seqs
    with open(gt_path, 'r') as f:
        f.readline() # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                seqs[int(parts[0])] = parts[1]
    return seqs

def find_best_match_for_chunk(chunk_preds, gt_seqs):
    """
    单个进程的任务：为分配给它的预测序列找到最佳匹配的 GT
    """
    results = []
    gt_items = list(gt_seqs.items()) 
    
    for pid, pseq in chunk_preds:
        best_id = -1
        best_score = -1.0
        
        # 优化：先检查 pid 是否直接对应 (假设 ID 对齐的情况)
        if pid in gt_seqs:
            direct_score = calculate_identity(pseq, gt_seqs[pid])
            if direct_score > 0.8: # 如果直接匹配已经很好了，剪枝
                results.append((pid, pid, direct_score, pseq, gt_seqs[pid]))
                continue
        
        # 否则全量搜索
        for gid, gseq in gt_items:
            score = calculate_identity(pseq, gseq)
            if score > best_score:
                best_score = score
                best_id = gid
            if score == 1.0: break 
            
        results.append((pid, best_id, best_score, pseq, gt_seqs[best_id]))
    return results

def main():
    # 直接从 CONFIG 读取参数，不再需要命令行
    print(f"⚙️  读取代码内置配置...")
    args = SimpleNamespace(**CONFIG)

    print(f"🚀 开始高性能验证 (Round {args.round})...")
    
    # 1. 路径构建
    consensus_path = os.path.join(args.result_dir, f"round_{args.round}", "step2", "consensus_sequences.fasta")
    print(f"📂 读取预测: {consensus_path}")
    
    if not os.path.exists(consensus_path):
        print(f"❌ 错误: 找不到预测文件: {consensus_path}")
        print("   请检查 CONFIG 中的 'result_dir' 和 'round' 是否正确")
        return

    pred_seqs = load_fasta(consensus_path)
    print(f"   - 预测序列数: {len(pred_seqs)}")
    
    print(f"📂 读取 GT: {args.exp_dir}")
    if not os.path.exists(args.exp_dir):
        print(f"❌ 错误: 找不到实验原始目录: {args.exp_dir}")
        return

    gt_seqs = load_gt_clusters(args.exp_dir)
    print(f"   - 真实序列数: {len(gt_seqs)}")
    
    if not pred_seqs or not gt_seqs:
        print("❌ 文件缺失或为空，无法验证")
        return

    # 2. 多进程并行计算
    print(f"\n⚡ 启动多进程匹配 (CPU核心数: {multiprocessing.cpu_count()})...")
    start_time = time.time()
    
    # 将任务分块
    pred_items = list(pred_seqs.items())
    num_processes = min(64, multiprocessing.cpu_count()) # 你的服务器有64核，直接拉满
    chunk_size = len(pred_items) // num_processes + 1
    chunks = [pred_items[i:i + chunk_size] for i in range(0, len(pred_items), chunk_size)]
    
    pool = multiprocessing.Pool(processes=num_processes)
    func = partial(find_best_match_for_chunk, gt_seqs=gt_seqs)
    
    all_matches = []
    # 使用 imap_unordered 可以显示进度条（如果加了的话），这里直接用 map 简单
    for res in pool.map(func, chunks):
        all_matches.extend(res)
        
    pool.close()
    pool.join()
    
    duration = time.time() - start_time
    print(f"✅ 匹配完成! 耗时: {duration:.1f}秒")

    # 3. 统计结果
    # （1） 统计完美匹配 (Precision角度，可能>GT)
    perfect_matches_count = sum(1 for m in all_matches if m[2] > 0.999)
    
    # （2） 统计唯一召回 (Recall角度，绝不会>GT)
    # 记录哪些 GT 被完美找回了
    recovered_gt_ids = set()
    for m in all_matches:
        if m[2] > 0.999:
            recovered_gt_ids.add(m[1]) # m[1] is GT_ID
                
    unique_recovered = len(recovered_gt_ids)
    recall = unique_recovered / len(gt_seqs)

    print("\n" + "="*40)
    print(f"🏆 Round {args.round} 最终验证结果")
    print(f"✅ 平均一致性 (Identity): {avg_identity:.2%}")
    print(f"✅ 预测簇总数: {len(all_matches)}")
    print(f"✅ 完美匹配数 (Precision-like): {perfect_matches_count}")
    print(f"🌟 唯一GT找回数 (Recall): {unique_recovered} / {len(gt_seqs)} ({recall:.2%})")
    print("="*40)
    
    # 4. 输出几个最好的样例
    all_matches.sort(key=lambda x: x[2], reverse=True)
    print("\n🔍 Top 3 样例:")
    for m in all_matches[:3]:
        print(f"Pred {m[0]} -> GT {m[1]} | Score: {m[2]:.2%}")

if __name__ == "__main__":
    main()