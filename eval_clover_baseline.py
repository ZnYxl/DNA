import os
import argparse
import numpy as np
import csv
from collections import Counter, defaultdict

def calculate_identity(seq1, seq2):
    """
    计算两条序列的同一性 (Identity)
    使用简单的编辑距离 (Levenshtein Distance)
    """
    if not seq1 or not seq2:
        return 0.0
    
    # 简单的 DP 计算编辑距离
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                           dp[i][j - 1] + 1,      # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution
                           
    distance = dp[m][n]
    max_len = max(len(seq1), len(seq2))
    return (1 - distance / max_len) * 100.0

class CloverEvaluator:
    def __init__(self, experiment_dir):
        self.exp_dir = experiment_dir
        self.raw_dir = os.path.join(experiment_dir, "01_RawData")
        self.feddna_dir = os.path.join(experiment_dir, "03_FedDNA_In")
        
        # 数据容器
        self.read_to_gt = {}     # Read_ID -> GT_Cluster_ID
        self.gt_to_seq = {}      # GT_Cluster_ID -> GT_Ref_Seq
        self.clover_clusters = defaultdict(list) # Clover_ID -> [Read_IDs]
        self.clover_centers = {} # Clover_ID -> Center_Sequence
        self.read_sequences = {} # Read_ID -> Sequence (用于反查)
        self.seq_to_id = {}      # Sequence -> Read_ID

    def load_ground_truth(self):
        """加载 GT 信息"""
        print("📂 1. 加载 Ground Truth...")
        
        # 1. Load Read GT
        gt_read_path = os.path.join(self.raw_dir, "ground_truth_reads.txt")
        if not os.path.exists(gt_read_path):
            print(f"   ⚠️ 找不到 GT 文件: {gt_read_path}，将跳过GT评估")
            return

        with open(gt_read_path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.read_to_gt[parts[0]] = int(parts[1])
        print(f"   - 加载了 {len(self.read_to_gt)} 条 Read GT")

        # 2. Load Cluster GT (Ref Seqs)
        gt_cluster_path = os.path.join(self.raw_dir, "ground_truth_clusters.txt")
        if os.path.exists(gt_cluster_path):
            with open(gt_cluster_path, 'r') as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        try:
                            self.gt_to_seq[int(parts[0])] = parts[1]
                        except ValueError:
                            continue
            print(f"   - 加载了 {len(self.gt_to_seq)} 条 GT 参考序列")

    def load_raw_reads_map(self):
        """加载原始 Reads 以便通过序列反查 ID"""
        print("📂 2. 建立序列到ID的映射...")
        raw_path = os.path.join(self.raw_dir, "raw_reads.txt")
        if not os.path.exists(raw_path):
            print(f"   ❌ 错误：找不到 raw_reads.txt: {raw_path}")
            return

        with open(raw_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.seq_to_id[parts[1]] = parts[0]
        print(f"   - 加载了 {len(self.seq_to_id)} 条原始序列映射")

    def load_clover_results(self):
        """加载 Clover 的聚类结果和中心序列"""
        print("📂 3. 加载 Clover 结果...")
        
        # 1. Load Clusters (read.txt)
        read_path = os.path.join(self.feddna_dir, "read.txt")
        if not os.path.exists(read_path):
            print(f"   ❌ 错误：找不到 Clover 输出文件 read.txt: {read_path}")
            return

        current_cluster = -1
        
        with open(read_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.startswith("====="):
                    current_cluster += 1
                else:
                    read_id = self.seq_to_id.get(line)
                    if read_id:
                        self.clover_clusters[current_cluster].append(read_id)
        
        print(f"   - 解析出 {len(self.clover_clusters)} 个 Clover 簇")

        # 2. Load Centers
        center_candidates = ["center.txt", "cluster_center.txt", "centers.txt"]
        center_path = None
        for name in center_candidates:
            p = os.path.join(self.feddna_dir, name)
            if os.path.exists(p):
                center_path = p
                break
        
        if center_path:
            idx = 0
            with open(center_path, 'r') as f:
                for line in f:
                    seq = line.strip()
                    if seq and not seq.startswith("==="):
                        self.clover_centers[idx] = seq
                        idx += 1
            print(f"   - 加载了 {len(self.clover_centers)} 条中心序列")
        else:
            print("   ⚠️ 未找到中心序列文件 (center.txt)，将跳过序列精度评估")

    def evaluate(self):
        print("\n" + "="*80)
        print("📊 Clover 基准线评估报告")
        print("="*80)
        
        if not self.clover_clusters:
            print("❌ 没有加载到任何聚类结果，请检查路径。")
            return [], 0, 0
            
        if not self.read_to_gt:
            print("⚠️ 没有加载到 GT，无法评估纯度。")
            return [], 0, 0
        
        total_purity = 0
        total_identity = 0
        valid_centers_count = 0
        valid_clusters_count = 0
        results = []
        
        print(f"{'Clover_ID':<10} | {'Dom_GT':<8} | {'Size':<6} | {'Purity(%)':<10} | {'Identity(%)':<12} | {'Note'}")
        print("-" * 80)
        
        for cid in sorted(self.clover_clusters.keys()):
            reads = self.clover_clusters[cid]
            if not reads: continue
            valid_clusters_count += 1
            
            gt_labels = [self.read_to_gt.get(r, -1) for r in reads]
            gt_counts = Counter(gt_labels)
            dominant_gt, dom_count = gt_counts.most_common(1)[0] if gt_counts else (-1, 0)
            
            purity = (dom_count / len(reads)) * 100.0
            total_purity += purity
            
            identity = 0.0
            note = ""
            
            if cid in self.clover_centers and dominant_gt in self.gt_to_seq:
                clover_seq = self.clover_centers[cid]
                gt_seq = self.gt_to_seq[dominant_gt]
                identity = calculate_identity(clover_seq, gt_seq)
                total_identity += identity
                valid_centers_count += 1
            elif dominant_gt == -1: note = "Noise Dominant"
            elif dominant_gt not in self.gt_to_seq: note = "GT Ref Missing"
            elif cid not in self.clover_centers: note = "No Center Seq"
            
            results.append({'Clover_ID': cid, 'Dominant_GT': dominant_gt, 'Size': len(reads), 'Purity': purity, 'Identity': identity, 'Note': note})
            
            if cid < 10 or purity < 90 or (identity > 0 and identity < 95):
                print(f"{cid:<10} | {dominant_gt:<8} | {len(reads):<6} | {purity:<10.1f} | {identity:<12.1f} | {note}")

        avg_purity = total_purity / valid_clusters_count if valid_clusters_count > 0 else 0
        avg_identity = total_identity / valid_centers_count if valid_centers_count > 0 else 0
        
        print("-" * 80)
        print(f"📈 总体平均纯度 (Avg Purity):   {avg_purity:.2f}% (基于 {valid_clusters_count} 个簇)")
        print(f"🎯 总体序列一致性 (Avg Identity): {avg_identity:.2f}% (基于 {valid_centers_count} 个匹配簇)")
        print("=" * 80)
        return results, avg_purity, avg_identity

    def save_csv(self, results, output_path):
        if not results: return
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Clover_ID', 'Dominant_GT', 'Size', 'Purity', 'Identity', 'Note'])
            writer.writeheader()
            writer.writerows(results)
        print(f"💾 详细报告已保存: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='clover_baseline.csv')
    args = parser.parse_args()
    
    evaluator = CloverEvaluator(args.experiment_dir)
    evaluator.load_ground_truth()
    evaluator.load_raw_reads_map()
    evaluator.load_clover_results()
    results, _, _ = evaluator.evaluate()
    evaluator.save_csv(results, args.output)
