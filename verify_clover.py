import os
import numpy as np
import glob
from collections import Counter, defaultdict
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
    """
    计算序列一致性 (Hamming 风格，简单匹配率)
    """
    if not seq1 or not seq2: return 0.0
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / max(len(seq1), len(seq2))

def find_best_match_chunk(chunk_data, gt_seqs):
    """
    多进程子任务：为一批 Clover 序列找最佳 GT
    chunk_data: list of (cid, c_seq)
    gt_seqs: dict {gid: g_seq}
    """
    results = []
    gt_items = list(gt_seqs.items())
    
    for cid, c_seq in chunk_data:
        best_score = 0.0
        best_gt_id = -1
        
        # 优化：先尝试直接对齐 (假设 ID 相同)
        if cid in gt_seqs:
            direct_score = calculate_identity(c_seq, gt_seqs[cid])
            if direct_score > 0.8: # 剪枝
                results.append({'clover_id': cid, 'gt_id': cid, 'identity': direct_score})
                continue

        # 全量搜索
        for gid, g_seq in gt_items:
            score = calculate_identity(c_seq, g_seq)
            if score > best_score:
                best_score = score
                best_gt_id = gid
            if score == 1.0: break
        
        results.append({'clover_id': cid, 'gt_id': best_gt_id, 'identity': best_score})
    
    return results

class CloverEvaluator:
    def __init__(self, experiment_dir):
        self.exp_dir = experiment_dir
        self.raw_dir = os.path.join(experiment_dir, "01_RawData")
        self.feddna_dir = os.path.join(experiment_dir, "03_FedDNA_In")
        
        self.read_to_gt = {}     
        self.gt_cluster_seqs = {} # GT ID -> 序列
        self.clover_clusters = defaultdict(list) 
        self.clover_centers = {}  # Clover ID -> 序列
        self.seq_to_ids = defaultdict(list)

    def load_ground_truth(self):
        print("📂 1. 加载 Ground Truth (GT)...")
        # 加载 Read GT
        gt_read_path = os.path.join(self.raw_dir, "ground_truth_reads.txt")
        if os.path.exists(gt_read_path):
            with open(gt_read_path, 'r') as f:
                f.readline()
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 2: self.read_to_gt[p[0]] = int(p[1])
        
        # 加载 Cluster GT 序列
        gt_cluster_path = os.path.join(self.raw_dir, "ground_truth_clusters.txt")
        if os.path.exists(gt_cluster_path):
            with open(gt_cluster_path, 'r') as f:
                f.readline()
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 2: 
                        try: self.gt_cluster_seqs[int(p[0])] = p[1]
                        except: continue
        print(f"   - GT统计: {len(self.read_to_gt)} 条 Reads, {len(self.gt_cluster_seqs)} 个 Clusters")

    def load_raw_reads_map(self):
        print("📂 2. 建立序列到 ID 的映射...")
        raw_path = os.path.join(self.raw_dir, "raw_reads.txt")
        if os.path.exists(raw_path):
            with open(raw_path, 'r') as f:
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 2: self.seq_to_ids[p[1]].append(p[0])

    def load_clover_results(self):
        print("📂 3. 加载 Clover 结果...")
        # 1. 加载 Reads
        read_path = os.path.join(self.feddna_dir, "read.txt")
        if os.path.exists(read_path):
            current_cluster = -1
            with open(read_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.startswith("====="):
                        current_cluster += 1
                    else:
                        pids = self.seq_to_ids.get(line)
                        if pids:
                            mid = next((i for i in pids if i in self.read_to_gt), pids[0])
                            self.clover_clusters[current_cluster].append(mid)

        # 2. 加载中心序列
        center_files = glob.glob(os.path.join(self.feddna_dir, "*ref*.txt")) + \
                       glob.glob(os.path.join(self.feddna_dir, "*center*.txt"))
        
        if center_files:
            target_file = center_files[0]
            print(f"   - 发现中心序列文件: {os.path.basename(target_file)}")
            
            with open(target_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('=')]
            
            has_headers = any(line.startswith('>') for line in lines)
            
            if has_headers:
                current_id = 0
                seq_buffer = []
                for line in lines:
                    if line.startswith('>'):
                        if seq_buffer:
                            self.clover_centers[current_id] = "".join(seq_buffer)
                            current_id += 1
                            seq_buffer = []
                        try:
                            parts = line.split('_')
                            if len(parts) > 1:
                                current_id = int(parts[1])
                        except:
                            pass
                    else:
                        seq_buffer.append(line)
                if seq_buffer:
                    self.clover_centers[current_id] = "".join(seq_buffer)
            else:
                print("   - 未检测到 Header，假设每行一条序列 (行号=ID)...")
                for idx, line in enumerate(lines):
                    if len(line) > 20: 
                        self.clover_centers[idx] = line

            print(f"   - 加载了 {len(self.clover_centers)} 条 Clover 生成的序列")
        else:
            print("   ⚠️ 未找到中心序列文件")

    def evaluate_identity_smart(self):
        """
        评估序列一致性 (多进程加速版)
        """
        print("\n📊 [Metric] Clover 序列一致性 (Pred -> GT Best Match)")
        
        if not self.clover_centers or not self.gt_cluster_seqs:
            print("   ⚠️ 缺少 GT 或 Clover 数据，跳过。")
            return

        print(f"⚡ 启动多进程匹配 (CPU核心数: {multiprocessing.cpu_count()})...")
        start_time = time.time()

        # 准备数据分块
        clover_items = list(self.clover_centers.items())
        num_processes = min(64, multiprocessing.cpu_count())
        chunk_size = len(clover_items) // num_processes + 1
        chunks = [clover_items[i:i + chunk_size] for i in range(0, len(clover_items), chunk_size)]

        # 并行计算
        pool = multiprocessing.Pool(processes=num_processes)
        func = partial(find_best_match_chunk, gt_seqs=self.gt_cluster_seqs)
        
        matches = []
        for res in pool.map(func, chunks):
            matches.extend(res)
        
        pool.close()
        pool.join()

        print(f"✅ 匹配完成! 耗时: {time.time() - start_time:.1f}秒")

        # 统计结果
        identities = [m['identity'] for m in matches]
        avg_identity = np.mean(identities)
        perfect_matches = sum(1 for x in identities if x > 0.99)
        
        matches_sorted = sorted(matches, key=lambda x: x['identity'], reverse=True)
        print("\n   🔍 Clover 最佳匹配样例 (Top 3):")
        for m in matches_sorted[:3]:
            print(f"   Clover {m['clover_id']} -> GT {m['gt_id']} | 一致性: {m['identity']:.2%}")

        print("\n" + "-"*40)
        print(f"🏆 Clover 真实基准 (Pred={len(matches)} vs GT={len(self.gt_cluster_seqs)})")
        print(f"✅ 平均一致性: {avg_identity:.2%}")
        print(f"✅ 完美匹配数: {perfect_matches}/{len(matches)}")
        print("-" * 40)

if __name__ == "__main__":
    # 直接实例化，无需 argparse
    print(f"🚀 开始评估实验: {EXPERIMENT_DIR}")
    eval = CloverEvaluator(EXPERIMENT_DIR)
    eval.load_ground_truth()
    eval.load_raw_reads_map()
    eval.load_clover_results()
    
    eval.evaluate_identity_smart()