import os
import difflib

# ================= 配置 =================
HARDCODED_CONFIG = {
    "gt_file": "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/20251224_155232_Cluster_GT_Test/01_RawData/ground_truth_clusters.txt",
    "pred_file": "/mnt/st_data/liangxinyi/code/iterative_results/20251224_155232_Cluster_GT_Test copy/round_3/step2/consensus_sequences_deduplicated.fasta",
    # 把刚才那 8 个 ID 填在这里
    "target_ids": [458, 886, 2236, 4163, 4810, 4963, 7532, 9946]
}
# =======================================

def calculate_identity(seq1, seq2):
    if not seq1 or not seq2: return 0.0
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / max(len(seq1), len(seq2))

def load_target_gt(path, target_ids):
    gt_map = {}
    with open(path, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                gid = int(parts[0])
                if gid in target_ids:
                    gt_map[gid] = parts[1]
    return gt_map

def load_all_preds(path):
    preds = []
    if not os.path.exists(path): return preds
    with open(path, 'r') as f:
        seq = []
        header = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq: preds.append((header, "".join(seq)))
                header = line
                seq = []
            else:
                seq.append(line)
        if seq: preds.append((header, "".join(seq)))
    return preds

def highlight_diff(seq1, seq2):
    """简单的差异高亮"""
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    diff = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            diff.append(seq1[a0:a1])
        elif opcode == 'insert':
            diff.append(f"[{seq2[b0:b1]}]") # 预测多出来的
        elif opcode == 'delete':
            diff.append(f"(-{seq1[a0:a1]})") # 预测漏掉的
        elif opcode == 'replace':
            diff.append(f"({seq1[a0:a1]}->{seq2[b0:b1]})") # 变异
    return "".join(diff)

def main():
    print("🚀 开始个案诊断...")
    gt_map = load_target_gt(HARDCODED_CONFIG['gt_file'], HARDCODED_CONFIG['target_ids'])
    all_preds = load_all_preds(HARDCODED_CONFIG['pred_file'])
    
    print(f"   已加载 {len(gt_map)} 个目标 GT，正在 {len(all_preds)} 个预测中搜索最佳匹配...")
    
    for gid, gseq in gt_map.items():
        best_score = 0.0
        best_pred_seq = ""
        best_pred_header = ""
        
        # 暴力搜索最佳匹配
        for ph, pseq in all_preds:
            score = calculate_identity(gseq, pseq)
            if score > best_score:
                best_score = score
                best_pred_seq = pseq
                best_pred_header = ph
        
        print(f"\n🔍 GT_ID: {gid} (Length: {len(gseq)})")
        print(f"   最佳匹配得分: {best_score:.4%}")
        
        if best_score > 0.99:
            print("   ✅ 结论: 这是一个【边界微错】案例 (Score > 99%)")
            print("      只需在论文中解释：'Minor indel/substitution errors slightly below threshold'.")
        else:
            print("   ❌ 结论: 这是一个【结构性困难】案例")
        
        # 打印差异
        # 如果长度差不多，展示差异
        if abs(len(gseq) - len(best_pred_seq)) < 20:
            print(f"   差异分析: {highlight_diff(gseq, best_pred_seq)}")
        else:
            print(f"   长度差异过大: GT={len(gseq)} vs Pred={len(best_pred_seq)}")

if __name__ == "__main__":
    main()