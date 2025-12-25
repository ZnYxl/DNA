# verify_final.py
import os
import numpy as np
import multiprocessing
from functools import partial
import time

# ==========================================
# 📝 【这里是你要修改的路径区域】
# ==========================================
HARDCODED_CONFIG = {
    # 1. 预测结果文件路径 (填那个未去重的文件即可，脚本会自动找去重版的)
    "pred_file": "/mnt/st_data/liangxinyi/code/iterative_results/20251224_155232_Cluster_GT_Test copy/round_3/step2/consensus_sequences.fasta",
    
    # 2. Ground Truth 所在的实验根目录
    "gt_dir": "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/20251224_155232_Cluster_GT_Test"
}
# ==========================================

def calculate_identity(seq1, seq2):
    """计算序列一致性"""
    if not seq1 or not seq2: return 0.0
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / max(len(seq1), len(seq2))

def load_fasta(path):
    """读取FASTA文件"""
    seqs = {}
    if not os.path.exists(path): return seqs
    with open(path, 'r') as f:
        header = None; seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # 解析Header中的ID
                try:
                    cluster_id = int(header.split('_')[1])
                except:
                    cluster_id = len(seqs) 
                
                if header: seqs[cluster_id] = "".join(seq)
                header = line; seq = []
            else: seq.append(line)
        if header: 
            try:
                cluster_id = int(header.split('_')[1])
            except:
                cluster_id = len(seqs)
            seqs[cluster_id] = "".join(seq)
    return seqs

def load_gt_clusters(experiment_dir):
    """加载Ground Truth"""
    gt_path = os.path.join(experiment_dir, "01_RawData", "ground_truth_clusters.txt")
    seqs = {}
    if not os.path.exists(gt_path): return seqs
    with open(gt_path, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                seqs[int(parts[0])] = parts[1]
    return seqs

def find_best_match_for_chunk(chunk_preds, gt_seqs):
    """多进程子任务"""
    results = []
    gt_items = list(gt_seqs.items()) 
    
    for pid, pseq in chunk_preds:
        best_id = -1
        best_score = -1.0
        
        # 1. 快速路径：检查相同ID
        if pid in gt_seqs:
            direct_score = calculate_identity(pseq, gt_seqs[pid])
            if direct_score > 0.8: 
                results.append((pid, pid, direct_score))
                continue
        
        # 2. 全盘扫描
        for gid, gseq in gt_items:
            score = calculate_identity(pseq, gseq)
            if score > best_score:
                best_score = score
                best_id = gid
            if score == 1.0: break 
            
        results.append((pid, best_id, best_score))
    return results

def main():
    print(f"⚙️  读取硬编码配置...")
    target_pred = HARDCODED_CONFIG['pred_file']
    gt_dir = HARDCODED_CONFIG['gt_dir']

    # 智能切换逻辑
    dedup_path = target_pred.replace(".fasta", "_deduplicated.fasta")
    
    if os.path.exists(dedup_path):
        print(f"✨ 检测到去重版文件存在，优先验证去重版！")
        final_pred_path = dedup_path
    else:
        print(f"⚠️ 未找到去重版，验证原始文件。")
        final_pred_path = target_pred

    print(f"📂 正在验证文件: {final_pred_path}")
    if not os.path.exists(final_pred_path):
        print(f"❌ 错误: 文件不存在 -> {final_pred_path}")
        return

    # 加载数据
    pred_seqs = load_fasta(final_pred_path)
    print(f"   - 预测簇数量: {len(pred_seqs)}")
    
    print(f"📂 读取 GT: {gt_dir}")
    gt_seqs = load_gt_clusters(gt_dir)
    print(f"   - 真实簇数量: {len(gt_seqs)}")
    
    if not pred_seqs or not gt_seqs:
        print("❌ 数据加载失败。")
        return

    # 多进程匹配
    num_cpus = min(64, multiprocessing.cpu_count())
    print(f"\n⚡ 启动多进程匹配 (CPU核心数: {num_cpus})...")
    start_time = time.time()
    
    pred_items = list(pred_seqs.items())
    chunk_size = len(pred_items) // num_cpus + 1
    chunks = [pred_items[i:i + chunk_size] for i in range(0, len(pred_items), chunk_size)]
    
    pool = multiprocessing.Pool(processes=num_cpus)
    func = partial(find_best_match_for_chunk, gt_seqs=gt_seqs)
    
    all_matches = []
    # 兼容 tqdm
    try:
        from tqdm import tqdm
        for res in tqdm(pool.imap_unordered(func, chunks), total=len(chunks), desc="匹配中"):
            all_matches.extend(res)
    except ImportError:
        for res in pool.map(func, chunks):
            all_matches.extend(res)
        
    pool.close()
    pool.join()
    
    print(f"✅ 匹配完成! 耗时: {time.time() - start_time:.1f}秒")

    # 统计指标
    avg_identity = np.mean([m[2] for m in all_matches])
    
    # Precision: 完美匹配的预测簇占比
    perfect_preds = sum(1 for m in all_matches if m[2] > 0.999)
    precision = perfect_preds / len(all_matches) if all_matches else 0

    # Recall: 唯一找回的GT占比
    recovered_gt_ids = set()
    for m in all_matches:
        if m[2] > 0.999:
            recovered_gt_ids.add(m[1]) 
            
    unique_recovered = len(recovered_gt_ids)
    recall = unique_recovered / len(gt_seqs)

    print("\n" + "="*60)
    print(f"🏆 最终验证结果")
    print("="*60)
    print(f"📂 验证文件: {os.path.basename(final_pred_path)}")
    print(f"📊 基础指标:")
    print(f"   - 预测簇数 (Pred): {len(all_matches)}")
    print(f"   - 真实簇数 (GT)  : {len(gt_seqs)}")
    print(f"   - 平均一致性 (Avg Identity): {avg_identity:.2%}")
    print("-" * 30)
    print(f"🎯 关键指标 (Strict > 99.9%):")
    print(f"   - 完美匹配数 (Perfect Matches): {perfect_preds}")
    print(f"   - 唯一GT找回数 (Unique GT Recovered): {unique_recovered}")
    print("-" * 30)
    print(f"🌟 最终得分:")
    print(f"   ✅ Precision (准确率/去噪能力): {precision:.2%}")
    print(f"   ✅ Recall    (召回率/恢复能力): {recall:.2%}")
    print("="*60)

if __name__ == "__main__":
    main()