"""
test_adaptive_zone.py - 验证自适应三区制划分

用已有的 read_state 数据, 对比 v2(硬编码) vs v3(自适应) 的划分结果。
不需要 GPU, 不需要重跑推理。

用法:
  cd /mnt/st_data/liangxinyi/code
  python test_adaptive_zone.py
"""
import os, sys, torch, numpy as np

CODE_DIR = "/mnt/st_data/liangxinyi/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

EXP_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Real_last"
LABELS_DIR = os.path.join(EXP_DIR, "04_Iterative_Labels")

# 三轮的时间戳 (231829=R1, 115135=R2, 201006=R3)
ROUNDS = [
    ("Round 1", "231829"),
    ("Round 2", "115135"),
    ("Round 3", "201006"),
]

# ---------------------------------------------------------------------------
# v2 硬编码版 (旧)
# ---------------------------------------------------------------------------
def split_v2(u_epi, u_ale, labels):
    DIRTY_P = 0.10
    SAFE_P  = 0.70

    N = len(labels)
    zone_ids = torch.zeros(N, dtype=torch.long)
    valid = (labels >= 0)

    ale_threshold = torch.quantile(u_ale[valid], 1.0 - DIRTY_P)
    is_dirty = valid & (u_ale >= ale_threshold)
    zone_ids[is_dirty] = 3

    remaining = valid & (~is_dirty)
    if remaining.any():
        epi_threshold = torch.quantile(u_epi[remaining], SAFE_P)
        zone_ids[remaining & (u_epi <= epi_threshold)] = 1
        zone_ids[remaining & (u_epi >  epi_threshold)] = 2

    return zone_ids, float(ale_threshold), float(epi_threshold)


# ---------------------------------------------------------------------------
# v3 自适应版 (新) - 从 step2_refine_v3.py 导入
# ---------------------------------------------------------------------------
# 把 v3 加到 path
sys.path.insert(0, '/home/claude')  # 如果从开发机跑
# 也尝试当前目录
sys.path.insert(0, '.')

try:
    from step2_refine_v3 import split_confidence_by_zone as split_v3
    print("✅ 从 step2_refine_v3 导入成功")
except ImportError:
    # fallback: 从 models 导入 (如果已替换)
    from models.step2_refine import split_confidence_by_zone as split_v3
    print("✅ 从 models.step2_refine 导入")


def main():
    for round_name, ts in ROUNDS:
        state_path = os.path.join(LABELS_DIR, f"read_state_{ts}.pt")
        label_path = os.path.join(LABELS_DIR, f"refined_labels_{ts}.txt")

        if not os.path.exists(state_path):
            print(f"\n⚠️ {round_name} state 不存在, 跳过")
            continue

        print(f"\n{'='*70}")
        print(f"🔬 {round_name} (ts={ts})")
        print(f"{'='*70}")

        state = torch.load(state_path, map_location='cpu')
        labels = torch.from_numpy(np.loadtxt(label_path, dtype=int))

        # state 里的数据可能是 numpy
        u_epi = state['u_epi']
        u_ale = state['u_ale']
        if isinstance(u_epi, np.ndarray):
            u_epi = torch.from_numpy(u_epi).float()
            u_ale = torch.from_numpy(u_ale).float()

        valid = (labels >= 0)
        n_valid = valid.sum().item()
        print(f"   有效 reads: {n_valid:,}")

        # --- v2 硬编码 ---
        print(f"\n   --- v2 硬编码 (P90 / P70) ---")
        z_v2, ale_t_v2, epi_t_v2 = split_v2(u_epi, u_ale, labels)
        z1_v2 = (z_v2 == 1).sum().item()
        z2_v2 = (z_v2 == 2).sum().item()
        z3_v2 = (z_v2 == 3).sum().item()
        print(f"   U_ale 阈值: {ale_t_v2:.6f}")
        print(f"   U_epi 阈值: {epi_t_v2:.6f}")
        print(f"   Zone I:  {z1_v2:>9,} ({z1_v2/n_valid*100:.1f}%)")
        print(f"   Zone II: {z2_v2:>9,} ({z2_v2/n_valid*100:.1f}%)")
        print(f"   Zone III:{z3_v2:>9,} ({z3_v2/n_valid*100:.1f}%)")

        # --- v3 自适应 ---
        print(f"\n   --- v3 自适应 (CDF knee + GMM) ---")
        z_v3, stats_v3 = split_v3(u_epi, u_ale, labels)
        z1_v3 = stats_v3['zone1']
        z2_v3 = stats_v3['zone2']
        z3_v3 = stats_v3['zone3']

        # --- 对比 ---
        print(f"\n   --- 对比 ---")
        print(f"   {'Zone':<10s} {'v2':>12s} {'v3':>12s} {'差异':>12s}")
        print(f"   {'─'*50}")
        for zname, v2, v3 in [("Zone I", z1_v2, z1_v3),
                               ("Zone II", z2_v2, z2_v3),
                               ("Zone III", z3_v2, z3_v3)]:
            diff = v3 - v2
            sign = "+" if diff > 0 else ""
            print(f"   {zname:<10s} {v2:>10,} ({v2/n_valid*100:.1f}%) "
                  f"{v3:>10,} ({v3/n_valid*100:.1f}%)  {sign}{diff:,}")

        # --- U_ale 分布统计 (帮助理解 CDF knee 的选择) ---
        ale_valid = u_ale[valid].numpy()
        print(f"\n   U_ale 分布: min={ale_valid.min():.6f}, "
              f"median={np.median(ale_valid):.6f}, "
              f"P90={np.quantile(ale_valid, 0.90):.6f}, "
              f"max={ale_valid.max():.6f}")

        epi_valid = u_epi[valid & (z_v2 != 3)].numpy()
        print(f"   U_epi 分布 (去Zone III): min={epi_valid.min():.6f}, "
              f"median={np.median(epi_valid):.6f}, "
              f"P70={np.quantile(epi_valid, 0.70):.6f}, "
              f"max={epi_valid.max():.6f}")

    print(f"\n{'='*70}")
    print("✅ 对比完成")


if __name__ == "__main__":
    main()