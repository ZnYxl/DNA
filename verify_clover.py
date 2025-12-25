# verify_clover.py
import os
import glob
import numpy as np
import multiprocessing
from functools import partial
import time

# ==========================================
# 📝 【配置区域】修改这里即可！
# ==========================================
# 你的实验路径
EXPERIMENT_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/20251224_155232_Cluster_GT_Test"
# ==========================================

def calculate_identity(seq1, seq2):
    """计算序列一致性 (SOTA标准)"""
    if not seq1 or not seq2: return 0.0
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / max(len(seq1), len(seq2))

def find_best_match_chunk(chunk_data, gt_seqs):
    """
    多进程子任务：为一批 Clover 序列找最佳 GT
    chunk_data: list of (clover_id, clover_seq)
    """
    results = []
    gt_items = list(gt_seqs.items())
    
    for cid, c_seq in chunk_data:
        best_score = 0.0
        best_gt_id = -1
        
        # 优化：虽然 Clover ID 通常不等于 GT ID，但万一撞上了呢？先试一下
        if cid in gt_seqs:
            direct_score = calculate_identity(c_seq, gt_seqs[cid])
            if direct_score > 0.8:
                results.append((cid, cid, direct_score))
                continue

        # 全量搜索 (Full Scan)
        for gid, g_seq in gt_items:
            # 简单长度过滤加速
            if abs(len(c_seq) - len(g_seq)) > 10: continue
            
            score = calculate_identity(c_seq, g_seq)
            if score > best_score:
                best_score = score
                best_gt_id = gid
            if score == 1.0: break
        
        results.append((cid, best_gt_id, best_score))
    
    return results

class CloverEvaluator:
    def __init__(self, experiment_dir):
        self.exp_dir = experiment_dir
        self.raw_dir = os.path.join(experiment_dir, "01_RawData")
        self.feddna_dir = os.path.join(experiment_dir, "03_FedDNA_In")
        self.gt_cluster_seqs = {} # GT ID -> 序列
        self.clover_centers = {}  # Clover ID -> 序列

    def load_ground_truth(self):
        print("📂 1. 加载 Ground Truth (GT)...")
        gt_path = os.path.join(self.raw_dir, "ground_truth_clusters.txt")
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                f.readline()
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 2: 
                        try: self.gt_cluster_seqs[int(p[0])] = p[1]
                        except: continue
        print(f"   - GT统计: {len(self.gt_cluster_seqs)} 个 Clusters")

    def load_clover_results(self):
        print("📂 2. 加载 Clover 结果...")
        # 寻找 Clover 输出的中心序列文件 (通常包含 'ref' 或 'center')
        center_files = glob.glob(os.path.join(self.feddna_dir, "*ref*.txt")) + \
                       glob.glob(os.path.join(self.feddna_dir, "*center*.txt"))
        
        if center_files:
            target_file = center_files[0]
            print(f"   - 发现中心序列文件: {os.path.basename(target_file)}")
            
            with open(target_file, 'r') as f:
                # 过滤掉空行和分割线
                lines = [line.strip() for line in f if line.strip() and not line.startswith('=')]
            
            has_headers = any(line.startswith('>') for line in lines)
            
            if has_headers:
                # 有 Header 的情况
                current_id = 0
                seq_buffer = []
                for line in lines:
                    if line.startswith('>'):
                        if seq_buffer:
                            self.clover_centers[current_id] = "".join(seq_buffer)
                            current_id += 1
                            seq_buffer = []
                        # 尝试从 header 解析 ID (如果格式允许)
                        try:
                            parts = line.split('_')
                            if len(parts) > 1 and parts[1].isdigit():
                                current_id = int(parts[1])
                        except: pass
                    else:
                        seq_buffer.append(line)
                if seq_buffer:
                    self.clover_centers[current_id] = "".join(seq_buffer)
            else:
                # 无 Header 的情况 (Clover 默认输出)
                print("   - 未检测到 Header，假设每行一条序列 (行号=ID)...")
                valid_count = 0
                for idx, line in enumerate(lines):
                    # 简单的长度过滤，防止读入垃圾数据
                    if len(line) > 20: 
                        self.clover_centers[idx] = line
                        valid_count += 1
            print(f"   - 加载了 {len(self.clover_centers)} 条 Clover 生成的序列")
        else:
            print("   ⚠️ 错误：未找到 Clover 中心序列文件！请检查 03_FedDNA_In 目录。")

    def evaluate(self):
        if not self.clover_centers or not self.gt_cluster_seqs:
            print("❌ 数据缺失，无法评估。")
            return

        print(f"\n⚡ 启动多进程匹配 (CPU核心数: {min(64, multiprocessing.cpu_count())})...")
        start_time = time.time()

        clover_items = list(self.clover_centers.items())
        num_processes = min(64, multiprocessing.cpu_count())
        chunk_size = len(clover_items) // num_processes + 1
        chunks = [clover_items[i:i + chunk_size] for i in range(0, len(clover_items), chunk_size)]

        pool = multiprocessing.Pool(processes=num_processes)
        func = partial(find_best_match_chunk, gt_seqs=self.gt_cluster_seqs)
        
        all_matches = []
        # 兼容 tqdm 进度条 (如果有安装)
        try:
            from tqdm import tqdm
            for res in tqdm(pool.imap_unordered(func, chunks), total=len(chunks), desc="匹配进度"):
                all_matches.extend(res)
        except ImportError:
            for res in pool.map(func, chunks):
                all_matches.extend(res)
        
        pool.close()
        pool.join()

        print(f"✅ 匹配完成! 耗时: {time.time() - start_time:.1f}秒")

        # ==========================================
        # 📊 核心统计部分 (完全对齐 verify_final)
        # ==========================================
        
        # 1. 基础指标
        avg_identity = np.mean([m[2] for m in all_matches])
        
        # 2. Precision (完美匹配的预测簇占比)
        perfect_preds = sum(1 for m in all_matches if m[2] > 0.999)
        precision = perfect_preds / len(all_matches) if all_matches else 0

        # 3. Recall (唯一找回的GT占比)
        recovered_gt_ids = set()
        for m in all_matches:
            if m[2] > 0.999:
                recovered_gt_ids.add(m[1]) # m[1] 是 GT_ID
        
        unique_recovered = len(recovered_gt_ids)
        total_gt = len(self.gt_cluster_seqs)
        recall = unique_recovered / total_gt if total_gt > 0 else 0

        # 4. 输出报告
        print("\n" + "="*60)
        print(f"🏆 Clover Baseline 验证结果")
        print("="*60)
        print(f"📊 基础指标:")
        print(f"   - 预测簇数 (Pred): {len(all_matches)}")
        print(f"   - 真实簇数 (GT)  : {total_gt}")
        print(f"   - 平均一致性 (Avg Identity): {avg_identity:.2%}")
        print("-" * 30)
        print(f"🎯 关键指标 (Strict > 99.9%):")
        print(f"   - 完美匹配数 (Perfect Matches): {perfect_preds}")
        print(f"   - 唯一GT找回数 (Unique GT Recovered): {unique_recovered}")
        print("-" * 30)
        print(f"🌟 最终得分:")
        print(f"   ✅ Precision (准确率): {precision:.2%}")
        print(f"   ✅ Recall    (召回率): {recall:.2%}")
        print("="*60)

if __name__ == "__main__":
    evaluator = CloverEvaluator(EXPERIMENT_DIR)
    evaluator.load_ground_truth()
    evaluator.load_clover_results()
    evaluator.evaluate()