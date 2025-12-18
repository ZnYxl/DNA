import os
import argparse
import numpy as np
import glob
from collections import Counter, defaultdict

# ==========================================
# 1. 核心计算函数 (与您的 run_loop.py 保持一致)
# ==========================================
def calculate_identity(seq1, seq2):
    """
    计算序列一致性 (Hamming 风格，简单匹配率)
    """
    if not seq1 or not seq2: return 0.0
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / max(len(seq1), len(seq2))

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
        # 1. 加载 Reads (用于计算纯度，虽然后面没打印，但保留逻辑)
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

        # 2. 加载中心序列 (用于计算 Identity) - 增强了解析逻辑
        center_files = glob.glob(os.path.join(self.feddna_dir, "*ref*.txt")) + \
                       glob.glob(os.path.join(self.feddna_dir, "*center*.txt"))
        
        if center_files:
            target_file = center_files[0]
            print(f"   - 发现中心序列文件: {os.path.basename(target_file)}")
            
            # 先读取所有非空行
            with open(target_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('=')]
            
            # 检测是否包含 FASTA 头 (>)
            has_headers = any(line.startswith('>') for line in lines)
            
            if has_headers:
                # 按照 FASTA 格式解析
                current_id = 0
                seq_buffer = []
                for line in lines:
                    if line.startswith('>'):
                        if seq_buffer:
                            self.clover_centers[current_id] = "".join(seq_buffer)
                            current_id += 1
                            seq_buffer = []
                        # 尝试从 Header 解析 ID (例如 >Cluster_12)
                        try:
                            parts = line.split('_')
                            if len(parts) > 1:
                                current_id = int(parts[1])
                        except:
                            pass
                    else:
                        seq_buffer.append(line)
                # 添加最后一条
                if seq_buffer:
                    self.clover_centers[current_id] = "".join(seq_buffer)
            else:
                # 按照纯序列解析 (每行一条)
                print("   - 未检测到 Header，假设每行一条序列 (行号=ID)...")
                for idx, line in enumerate(lines):
                    # 过滤过短的噪声
                    if len(line) > 20: 
                        self.clover_centers[idx] = line

            print(f"   - 加载了 {len(self.clover_centers)} 条 Clover 生成的序列")
        else:
            print("   ⚠️ 未找到中心序列文件")

    def evaluate_identity_smart(self):
        """
        评估序列一致性 (以预测为中心)
        遍历每一个 Clover 预测结果 -> 去 GT 里找最佳匹配 (Best Match)
        分母 = Clover 预测出的簇数量
        """
        print("\n📊 [Metric] Clover 序列一致性 (Pred -> GT Best Match)")
        
        if not self.clover_centers or not self.gt_cluster_seqs:
            print("   ⚠️ 缺少 GT 或 Clover 数据，跳过。")
            return

        matches = []
        
        # 遍历 Clover 的预测结果
        for cid, c_seq in self.clover_centers.items():
            best_score = 0.0
            best_gt_id = -1
            
            # 在 GT 中寻找最相似的
            for gid, g_seq in self.gt_cluster_seqs.items():
                score = calculate_identity(c_seq, g_seq)
                if score > best_score:
                    best_score = score
                    best_gt_id = gid
            
            matches.append({
                'clover_id': cid,
                'gt_id': best_gt_id,
                'identity': best_score
            })

        # 统计结果
        identities = [m['identity'] for m in matches]
        avg_identity = np.mean(identities)
        perfect_matches = sum(1 for x in identities if x > 0.99)
        
        # 打印前3名最佳匹配
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    args = parser.parse_args()
    
    eval = CloverEvaluator(args.experiment_dir)
    eval.load_ground_truth()
    eval.load_raw_reads_map()
    eval.load_clover_results()
    
    eval.evaluate_identity_smart()