import os
import argparse

# 纯 Python 实现的序列比对，不需要额外依赖
def calculate_identity(seq1, seq2):
    if not seq1 or not seq2: return 0.0
    m, n = len(seq1), len(seq2)
    # 初始化 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    # 计算编辑距离
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                           dp[i][j - 1] + 1,      # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution
    
    # 转换为一致性百分比
    distance = dp[m][n]
    max_len = max(len(seq1), len(seq2))
    return (1 - distance / max_len) * 100.0

def check_identity_match(exp_dir):
    print(f"📂 正在分析实验目录: {exp_dir}")
    
    gt_path = os.path.join(exp_dir, "01_RawData", "ground_truth_clusters.txt")
    ref_path = os.path.join(exp_dir, "03_FedDNA_In", "ref.txt")
    
    if not os.path.exists(gt_path):
        print(f"❌ 找不到 GT 文件: {gt_path}")
        return
    if not os.path.exists(ref_path):
        print(f"❌ 找不到 ref.txt 文件: {ref_path}")
        return

    # 1. 加载所有 GT 序列
    gt_seqs = {}
    with open(gt_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    gt_seqs[int(parts[0])] = parts[1]
                except ValueError:
                    continue
                
    # 2. 加载 Clover 输出的 ref.txt (忽略顺序，全部存进列表)
    clover_seqs = []
    with open(ref_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 过滤掉 FASTA 头 (>Cluster_1) 或 分隔符 (=====)
            if not line or line.startswith(">") or line.startswith("="):
                continue
            if len(line) > 30: # 假设有效序列长度至少 30bp
                clover_seqs.append(line)
                
    print(f"📊 数据统计:")
    print(f"   - GT 参考序列数: {len(gt_seqs)}")
    print(f"   - Clover 输出序列数: {len(clover_seqs)}")
    
    if len(clover_seqs) == 0:
        print("❌ 错误: ref.txt 里好像没读到有效序列，请检查文件格式。")
        return

    # 3. 最佳匹配测试 (Best Match Search)
    print("\n🔄 正在进行乱序最佳匹配搜索 (这可能需要几秒钟)...")
    total_best_identity = 0
    match_details = []
    
    for gt_id, gt_seq in gt_seqs.items():
        best_score = 0
        best_match_seq = ""
        
        # 拿这条 GT 去跟所有的 Clover 序列比，找最像的那个
        for cl_seq in clover_seqs:
            score = calculate_identity(gt_seq, cl_seq)
            if score > best_score:
                best_score = score
                best_match_seq = cl_seq
        
        total_best_identity += best_score
        match_details.append(best_score)
        
        # 打印部分低分结果，方便诊断
        if best_score < 90:
            print(f"   ⚠️ GT Cluster {gt_id} 匹配度较低: {best_score:.2f}%")

    avg_best = total_best_identity / len(gt_seqs)
    
    print("\n" + "="*50)
    print(f"✅ 修正顺序后的真实平均一致性: {avg_best:.2f}%")
    if avg_best > 98:
        print("🎉 结论: Clover 其实跑得很准啊！")
    elif avg_best < 80:
        print("🤔 结论: Clover 生成的序列确实质量不高。")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    args = parser.parse_args()
    check_identity_match(args.experiment_dir)
