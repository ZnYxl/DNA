import os
import sys
import torch
import glob
from types import SimpleNamespace
import numpy as np

# 导入你的模块
from models.step1_train import train_step1
from models.step2_runner import run_step2
from models.step1_data import CloverDataLoader

# ================= 配置区域 =================
# 1. 只需要修改这里的输入路径，输出路径会自动跟随
INPUT_EXP_DIR = "CC/Step0/Experiments/20251224_155232_Cluster_GT_Test"

# 自动提取文件夹名称 (e.g., "20251218_231311_Cluster_GT_Test")
EXP_NAME = os.path.basename(os.path.normpath(INPUT_EXP_DIR))

CONFIG = {
    "experiment_dir": INPUT_EXP_DIR,
    "feddna_checkpoint": "result/FLDNA_I/I_1214234233/model/epoch1_I.pth",
    
    # ✅ 修改点1：输出目录自动带上时间戳
    "base_output_dir": os.path.join("./iterative_results", EXP_NAME),
    
    "max_rounds": 3,
    "device": "cuda",
    "epochs": 15,       # 大数据量下，15轮通常足够，30轮可能太久
    
    # ✅ 修改点2：针对百万级数据，必须增大 Batch Size
    "batch_size": 512,  # 建议 512 或 1024
    
    "lr": 1e-4
}
# ===========================================

def calculate_identity(seq1, seq2):
    """计算序列一致性"""
    if not seq1 or not seq2: return 0.0
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / max(len(seq1), len(seq2))

def verify_accuracy_smart(consensus_file, gt_data_loader):
    """
    ⚠️ 注意：对于 10,000 个簇，这个函数的运行时间会非常长（O(N^2)复杂度）。
    如果是百万级数据实验，建议先跳过此步骤，或者只在最终轮次离线运行。
    """
    if not os.path.exists(consensus_file): return

    print(f"\n📊 [Verify] 智能验证准确度 (Best Match Mode)...")
    
    # 1. 读取预测序列
    pred_seqs = {}
    with open(consensus_file, 'r') as f:
        header = None; seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header: pred_seqs[int(header.split('_')[1])] = "".join(seq)
                header = line; seq = []
            else: seq.append(line)
        if header: pred_seqs[int(header.split('_')[1])] = "".join(seq)

    # 2. 获取 GT
    gt_seqs = gt_data_loader.gt_cluster_seqs
    if not gt_seqs: return

    # 简单跳过检查：如果簇太多，为了防止卡死，只验证前 100 个 (可选)
    # 如果你想全量验证，请注释掉下面这两行
    if len(pred_seqs) > 2000:
        print(f"   ⚠️ 簇数量过大 ({len(pred_seqs)})，为节省时间，本次迭代跳过全量验证。")
        return

    # 3. 寻找最佳匹配 (Greedy Best Match)
    matches = []
    
    # 对每个预测簇，去 GT 里找一个最像的
    for pid, pseq in pred_seqs.items():
        best_id = -1
        best_score = -1.0
        
        for gid, gseq in gt_seqs.items():
            score = calculate_identity(pseq, gseq)
            if score > best_score:
                best_score = score
                best_id = gid
        
        matches.append({
            'pred_id': pid,
            'gt_id': best_id,
            'identity': best_score,
            'pred_seq': pseq,
            'gt_seq': gt_seqs[best_id]
        })

    # 4. 统计结果
    avg_identity = np.mean([m['identity'] for m in matches])
    perfect_matches = sum(1 for m in matches if m['identity'] > 0.99)
    
    print("\n   🔍 最佳匹配样例 (Top 3):")
    for m in sorted(matches, key=lambda x: x['identity'], reverse=True)[:3]:
        print(f"   Pred {m['pred_id']} -> GT {m['gt_id']} | Identity: {m['identity']:.2%}")
        print(f"     GT  : {m['gt_seq'][:30]}...")
        print(f"     PRED: {m['pred_seq'][:30]}...")

    print("\n" + "-"*40)
    print(f"🏆 真实验证结果 (校正ID后)")
    print(f"✅ 平均一致性: {avg_identity:.2%}")
    print(f"✅ 完美匹配数: {perfect_matches}/{len(matches)}")
    print("-"*40 + "\n")

def run_loop():
    print(f"🚀 开始 Python 自动迭代训练")
    print(f"📂 输入目录: {CONFIG['experiment_dir']}")
    print(f"📂 输出目录: {CONFIG['base_output_dir']}")
    print(f"⚙️  Batch Size: {CONFIG['batch_size']}")
    
    prev_labels = None
    current_checkpoint = CONFIG['feddna_checkpoint']
    
    # 加载 GT 数据 (如果文件很大，这一步可能会花点时间)
    print("📂 尝试加载 GT 数据...")
    try: gt_loader = CloverDataLoader(CONFIG['experiment_dir'])
    except: gt_loader = None

    for round_idx in range(1, CONFIG['max_rounds'] + 1):
        print(f"\n{'='*50}\n🔄 Round {round_idx} / {CONFIG['max_rounds']}\n{'='*50}")

        round_dir = os.path.join(CONFIG['base_output_dir'], f"round_{round_idx}")
        step1_out = os.path.join(round_dir, "step1")
        step2_out = os.path.join(round_dir, "step2")
        os.makedirs(step1_out, exist_ok=True)

        # Step 1: 训练
        # 注意：这里我们把 batch_size 传进去
        args_s1 = SimpleNamespace(
            experiment_dir=CONFIG['experiment_dir'],
            output_dir=step1_out,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            lr=CONFIG['lr'],
            weight_decay=1e-5,
            device=CONFIG['device'],
            dim=256, max_length=150, min_clusters=50, max_clusters_per_batch=5,
            save_interval=100,
            feddna_checkpoint=current_checkpoint,
            refined_labels=prev_labels
        )
        print("▶️  Running Step 1 (Training)...")
        model_path = train_step1(args_s1)
        current_checkpoint = model_path

        # Step 2: 推理与修正
        args_s2 = SimpleNamespace(
            experiment_dir=CONFIG['experiment_dir'],
            step1_checkpoint=model_path,
            output_dir=step2_out,
            uncertainty_percentile=0.2, delta=None, delta_percentile=10,
            dim=256, max_length=150, device=CONFIG['device']
        )
        print("▶️  Running Step 2 (Refining)...")
        run_step2(args_s2)
        
        # Verify: 验证
        # 我在 verify_accuracy_smart 里加了保护逻辑，如果簇太多会自动跳过
        if gt_loader:
            verify_accuracy_smart(os.path.join(step2_out, "consensus_sequences.fasta"), gt_loader)

        # Next Round: 准备下一轮标签
        label_dir = os.path.join(CONFIG['experiment_dir'], "04_Iterative_Labels")
        files = glob.glob(os.path.join(label_dir, 'refined_labels_*.txt'))
        if files: 
            # 找到最新的标签文件
            prev_labels = max(files, key=os.path.getctime)
            print(f"🔄 下一轮将使用标签: {os.path.basename(prev_labels)}")
        else: 
            print("❌ 未找到新生成的标签文件，迭代停止。")
            break

if __name__ == "__main__":
    run_loop()