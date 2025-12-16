import os

# ================= 配置区域 =================
# 1. 模型修正后的结果 (Step 2 的输出)
# 通常是 Dataset 下面的 ref.txt
PRED_REF_PATH = "ref_corrected.txt"

# 2. 真值文件 (Ground Truth)
# 【重要】请去 CC/Step0/Experiments/ 里找你刚才跑的那次实验的 ground_truth.txt
# 示例: "CC/Step0/Experiments/20251216_xxxxxx_Improved_Data_Test/01_RawData/ground_truth.txt"
GT_PATH = "CC/Step0/Experiments/20251216_145746_Improved_Data_Test/01_RawData/ground_truth.txt" 
# (如果不确定路径，请先在终端用 find 命令找一下)
# ===========================================

def calculate_accuracy():
    print(f"🚀 开始验证结果...")
    print(f"📥 预测文件: {PRED_REF_PATH}")
    print(f"📥 真值文件: {GT_PATH}")
    
    if not os.path.exists(PRED_REF_PATH) or not os.path.exists(GT_PATH):
        print("❌ 错误: 找不到文件，请检查路径配置！")
        return

    # 1. 加载真值 (Cluster_ID -> Ref_Seq)
    gt_refs = {}
    with open(GT_PATH, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # 假设格式: ReadID, ClusterID, RefSeq...
                c_id = parts[1]
                seq = parts[2]
                gt_refs[c_id] = seq # 记录每个簇对应的真实序列
    
    # 获取唯一的真值序列 (因为 GT 文件里每条 Read 都有 Ref，其实是重复的)
    # 我们只关心每个 Cluster ID 对应的真值是什么
    unique_gt = {}
    for cid, seq in gt_refs.items():
        unique_gt[cid] = seq
        
    print(f"✅ 加载真值: 共 {len(unique_gt)} 个簇的参考序列")

    # 2. 加载预测值 (行号/顺序 -> Seq)
    # 假设 ref.txt 是按 Cluster 0, 1, 2... 顺序排列的
    pred_refs = []
    with open(PRED_REF_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("="):
                pred_refs.append(line)
    
    print(f"✅ 加载预测: 共 {len(pred_refs)} 条序列")

    # 3. 比对计算 (这里做个简化假设：预测的第 i 行对应真值的 Cluster i)
    # 如果你的 Cluster ID 不是 0,1,2... 顺序，可能需要更复杂的对齐
    
    total_score = 0
    perfect_matches = 0
    count = 0
    
    # 把 GT 的 key 转成 list 方便索引 (假设是排好序的)
    gt_keys = sorted(unique_gt.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    min_len = min(len(pred_refs), len(gt_keys))
    
    print("\n🔍 详细比对:")
    for i in range(min_len):
        p_seq = pred_refs[i]
        gt_id = gt_keys[i]
        g_seq = unique_gt[gt_id]
        
        # 计算相似度 (1 - 汉明距离/长度)
        dist = sum(c1 != c2 for c1, c2 in zip(p_seq, g_seq))
        # 处理长度不一致的情况
        len_diff = abs(len(p_seq) - len(g_seq))
        dist += len_diff
        
        max_len = max(len(p_seq), len(g_seq))
        accuracy = 1.0 - (dist / max_len)
        
        total_score += accuracy
        if dist == 0:
            perfect_matches += 1
            
        count += 1
        print(f"   Cluster {gt_id}: Acc = {accuracy*100:.2f}%")
        
    avg_acc = total_score / count
    perfect_rate = perfect_matches / count
    
    print("-" * 40)
    print(f"🏆 最终成绩单:")
    print(f"   平均准确率 (Base Accuracy): {avg_acc*100:.2f}%")
    print(f"   完美恢复率 (Perfect Cluster): {perfect_rate*100:.2f}% ({perfect_matches}/{count})")
    print("-" * 40)
    
    if avg_acc > 0.95:
        print("🌟 教授评语: 完美！这绝对是顶刊水平！")
    elif avg_acc > 0.85:
        print("👍 教授评语: 很棒！显著优于传统算法。")
    else:
        print("💪 教授评语: 还有提升空间，可能是对齐没对上，或者噪声太大了。")

if __name__ == "__main__":
    calculate_accuracy()