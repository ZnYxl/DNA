import argparse
import os
import sys
import numpy as np

# 这一步是为了能 import 你的 models 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.step1_data import CloverDataLoader

def load_fasta(fasta_path):
    """简单的 FASTA 读取器"""
    sequences = {}
    current_header = None
    current_seq = []
    
    if not os.path.exists(fasta_path):
        print(f"❌ File not found: {fasta_path}")
        return {}

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    sequences[current_header] = "".join(current_seq)
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            sequences[current_header] = "".join(current_seq)
    
    # 解析 Header 获取 Cluster ID
    # Header 格式: >cluster_0_reads49...
    parsed_seqs = {}
    for header, seq in sequences.items():
        try:
            # 提取 cluster_ID
            parts = header.split('_')
            # 假设格式固定为 >cluster_X_...
            if parts[0] == ">cluster":
                cluster_id = int(parts[1])
                parsed_seqs[cluster_id] = seq
        except Exception as e:
            print(f"⚠️ Warning: Could not parse header {header}: {e}")
            continue
            
    return parsed_seqs

def calculate_identity(seq1, seq2):
    """计算两个序列的一致性 (简单 Hamming 变种，假设长度对齐或取最小)"""
    # 如果你生成的序列长度和 GT 不一样，可能需要 Needleman-Wunsch 算法
    # 这里为了简单，我们假设 Step 1 已经把长度 Pad 到了 150，或者我们只比较重叠部分
    
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    if min_len == 0:
        return 0.0
        
    matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
    # 惩罚长度差异
    identity = matches / max_len
    return identity

def levenshtein_distance(s1, s2):
    """计算编辑距离 (Levenshtein)"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def main():
    parser = argparse.ArgumentParser(description="Verify Reconstruction Accuracy")
    parser.add_argument('--experiment_dir', type=str, required=True, help="Data directory containing GT")
    parser.add_argument('--consensus_file', type=str, required=True, help="Generated consensus.fasta")
    args = parser.parse_args()

    print(f"\n🔍 Verifying Accuracy...")
    print(f"   GT Source: {args.experiment_dir}")
    print(f"   Consensus: {args.consensus_file}")

    # 1. 加载 Ground Truth
    # 利用现有的 DataLoader 加载 GT
    try:
        loader = CloverDataLoader(args.experiment_dir)
        gt_dict = loader.gt_cluster_seqs # {cluster_id: sequence}
        
        if not gt_dict:
            print("❌ No Ground Truth clusters found in experiment dir!")
            return
            
        print(f"   ✅ Loaded {len(gt_dict)} GT sequences.")
    except Exception as e:
        print(f"❌ Failed to load GT: {e}")
        return

    # 2. 加载预测的 Consensus
    pred_dict = load_fasta(args.consensus_file)
    print(f"   ✅ Loaded {len(pred_dict)} Predicted consensus sequences.")

    # 3. 比对
    metrics = {
        'identities': [],
        'edit_distances': [],
        'perfect_matches': 0,
        'missing_clusters': 0
    }
    
    print("\n   📊 Detailed Comparison (Top 5 examples):")
    print("   " + "-"*60)
    
    count = 0
    for cid, gt_seq in gt_dict.items():
        if cid in pred_dict:
            pred_seq = pred_dict[cid]
            
            # 计算指标
            ident = calculate_identity(pred_seq, gt_seq)
            edit_dist = levenshtein_distance(pred_seq, gt_seq)
            
            metrics['identities'].append(ident)
            metrics['edit_distances'].append(edit_dist)
            
            if ident == 1.0 and len(pred_seq) == len(gt_seq):
                metrics['perfect_matches'] += 1
                
            # 打印前几个例子
            if count < 5:
                print(f"   Cluster {cid}:")
                print(f"     GT  : {gt_seq[:50]}... (len={len(gt_seq)})")
                print(f"     PRED: {pred_seq[:50]}... (len={len(pred_seq)})")
                print(f"     -> Identity: {ident:.2%}, Edit Dist: {edit_dist}")
                count += 1
        else:
            metrics['missing_clusters'] += 1

    # 4. 汇总报告
    num_compared = len(metrics['identities'])
    if num_compared == 0:
        print("\n❌ No common clusters found between GT and Prediction.")
        return

    avg_identity = sum(metrics['identities']) / num_compared
    avg_edit_dist = sum(metrics['edit_distances']) / num_compared
    perfect_rate = metrics['perfect_matches'] / num_compared

    print("\n" + "="*60)
    print("🏆 Verification Results")
    print("="*60)
    print(f"   Compared Clusters : {num_compared}")
    print(f"   Missing Clusters  : {metrics['missing_clusters']}")
    print(f"   ---------------------------")
    print(f"   ✅ Average Identity  : {avg_identity:.2%}  (Target: >99%)")
    print(f"   ✅ Avg Edit Distance : {avg_edit_dist:.2f}    (Target: <1.0)")
    print(f"   ✅ Perfect Matches   : {metrics['perfect_matches']} ({perfect_rate:.1%})")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()