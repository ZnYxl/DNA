#!/usr/bin/env python
"""
spike_e_heavy.py —— Path E 重量 spike: frozen R1 encoder + iterate R2'/R3'

设计:
  - 完全独立目录, 不污染 v19 现有数据
  - frozen R1 encoder: R2' 和 R3' 都加载 R1 checkpoint, 不重训
  - 其他参数与 v19 完全一致 (rebirth=nearest, fasta_source=mv_strict 等)
  - 跑 2 轮 (R2', R3'), 5-6 小时

目录:
  v19_dir/                  (read-only 输入源)
    ├── 03_FedDNA_In/
    ├── 04_Iterative_Labels/refined_labels_<R1_ts>.txt 等
    └── results/iter_1_step1/models/step1_final_model.pth

  spike_dir/                (输出, 完全独立)
    ├── 03_FedDNA_In/       (软链 v19)
    ├── results/
    │   ├── iter_1_step1/   (软链 v19, 让 ckpt 路径统一)
    │   ├── iter_2_step2/   (R2' 输出)
    │   └── iter_3_step2/   (R3' 输出)
    └── 04_Iterative_Labels/
        ├── refined_labels_<R1_ts>.txt  等 (复制自 v19)
        ├── refined_labels_<R2'_ts>.txt 等 (本次生成)
        └── refined_labels_<R3'_ts>.txt 等

关键改动 (vs v19): step1_checkpoint 在 R2/R3 永远使用 R1 的, 不重训.

后续评估: 用 v19 同样的 eval_reconstruction.py 跑 R2'.fasta 和 R3'.fasta,
          得到 SR, 与 R1 SR (0.9139) 比较.
"""
import os, sys, glob, shutil, argparse, time


def find_r1_files(v19_dir):
    """从 v19 目录自动发现 R1 (最早 timestamp) 的所有文件"""
    labels_dir = os.path.join(v19_dir, '04_Iterative_Labels')
    label_files = sorted(
        glob.glob(os.path.join(labels_dir, 'refined_labels_*.txt')),
        key=os.path.getmtime
    )
    if not label_files:
        return None

    r1_labels = label_files[0]
    ts = os.path.basename(r1_labels).replace(
        'refined_labels_', '').replace('.txt', '')

    files = {
        'ts': ts,
        'labels': r1_labels,
        'state': os.path.join(labels_dir, f'read_state_{ts}.pt'),
        'centroids': os.path.join(labels_dir, f'centroids_{ts}.pt'),
        'consensus': os.path.join(labels_dir, f'consensus_dict_{ts}.pt'),
    }

    missing = [k for k, v in files.items()
               if k != 'ts' and not os.path.exists(v)]
    if missing:
        print(f"  ❌ R1 缺失: {missing}")
        return None

    # checkpoint (兼容两种路径)
    candidates = [
        os.path.join(v19_dir, 'results', 'iter_1_step1', 'models',
                     'step1_final_model.pth'),
        os.path.join(v19_dir, 'results', 'iter_1_step1',
                     'step1_final_model.pth'),
    ]
    ckpt = next((c for c in candidates if os.path.exists(c)), None)
    if not ckpt:
        print(f"  ❌ R1 checkpoint 找不到, 试过: {candidates}")
        return None
    files['checkpoint'] = ckpt
    return files


def setup_spike_dir(v19_dir, spike_dir, r1):
    """准备独立 spike 目录, 软链大目录 + 复制 R1 状态文件"""
    os.makedirs(spike_dir, exist_ok=True)

    # 1. 软链 03_FedDNA_In/ (~GB, 不复制)
    src = os.path.join(v19_dir, '03_FedDNA_In')
    dst = os.path.join(spike_dir, '03_FedDNA_In')
    if not os.path.exists(dst) and os.path.exists(src):
        os.symlink(src, dst)
        print(f"  🔗 软链 03_FedDNA_In/")

    # 2. 复制 R1 状态文件到 04_Iterative_Labels/
    dst_labels = os.path.join(spike_dir, '04_Iterative_Labels')
    os.makedirs(dst_labels, exist_ok=True)
    for k in ['labels', 'state', 'centroids', 'consensus']:
        src = r1[k]
        dst = os.path.join(dst_labels, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(src) / 1024**2
            print(f"  📋 复制 R1 {k:>10}: {os.path.basename(src)} "
                  f"({size_mb:.1f} MB)")

    # 3. 软链 results/iter_1_step1/ (含 checkpoint)
    src = os.path.join(v19_dir, 'results', 'iter_1_step1')
    dst = os.path.join(spike_dir, 'results', 'iter_1_step1')
    if not os.path.exists(dst) and os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)
        print(f"  🔗 软链 results/iter_1_step1/")


def main():
    p = argparse.ArgumentParser(description="Path E heavy spike")
    p.add_argument('--v19_dir', required=True,
                   help='v19 实验目录 (read-only)')
    p.add_argument('--spike_dir', required=True,
                   help='spike 输出目录 (独立)')
    p.add_argument('--gt_tags_file', required=True)
    p.add_argument('--gt_refs_file', required=True)
    p.add_argument('--feddna_checkpoint', required=True)
    p.add_argument('--code_dir', default='/mnt/st_data/liangxinyi/code',
                   help='项目根目录 (含 models/)')
    p.add_argument('--n_extra_rounds', type=int, default=2,
                   help='跑几轮额外 (R2, R3 → 2)')

    # v19 参数 (默认锁定与 v19 一致)
    p.add_argument('--target_clusters', type=int, default=11736)
    p.add_argument('--ref_length', type=int, default=196)
    p.add_argument('--max_length', type=int, default=201)
    p.add_argument('--primer_prefix', type=int, default=20)
    p.add_argument('--primer_suffix', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--dim', type=int, default=256)
    args = p.parse_args()

    t_start = time.time()

    # ── 1. 发现 R1 文件 ──────────────────────────────────────────────
    print("=" * 70)
    print("🔍 发现 R1 文件 (v19)")
    print("=" * 70)
    r1 = find_r1_files(args.v19_dir)
    if not r1:
        print("❌ R1 文件不全, 退出"); sys.exit(1)
    print(f"  R1 timestamp:  {r1['ts']}")
    print(f"  R1 checkpoint: {r1['checkpoint']}")
    print(f"  R1 labels:     {os.path.basename(r1['labels'])}")
    print(f"  R1 state:      {os.path.basename(r1['state'])}")
    print(f"  R1 centroids:  {os.path.basename(r1['centroids'])}")
    print(f"  R1 consensus:  {os.path.basename(r1['consensus'])}")

    # ── 2. 准备独立 spike 目录 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("📦 准备独立 spike 目录")
    print("=" * 70)
    print(f"  目标: {args.spike_dir}")
    setup_spike_dir(args.v19_dir, args.spike_dir, r1)

    # ── 3. 延迟 import step2_runner (避免 setup 阶段触发 GPU 初始化) ──
    if args.code_dir not in sys.path:
        sys.path.insert(0, args.code_dir)
    from models.step2_runner import run_step2

    # ── 4. 迭代跑 R2', R3', ... ─────────────────────────────────────
    current_labels    = r1['labels']
    current_state     = r1['state']
    current_consensus = r1['consensus']

    history = [{'round': 'R1 (v19 baseline)',
                'labels': r1['labels'], 'fasta': None}]

    for round_idx in range(2, 2 + args.n_extra_rounds):
        print("\n" + "=" * 70)
        print(f"🔁 Round {round_idx}' (frozen R1 encoder)")
        print(f"   input labels: {os.path.basename(current_labels)}")
        print("=" * 70)

        step2_out = os.path.join(args.spike_dir, 'results',
                                 f'iter_{round_idx}_step2')
        os.makedirs(step2_out, exist_ok=True)

        step2_args = argparse.Namespace(
            # ★ KEY: 每轮都用 R1 checkpoint
            step1_checkpoint=r1['checkpoint'],

            experiment_dir=args.spike_dir,
            output_dir=step2_out,
            round_idx=round_idx,
            refined_labels=current_labels,
            prev_state=current_state,
            consensus_path=current_consensus,

            # v19 参数 (锁定一致)
            dim=args.dim,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device='cuda',
            training_cap=9999000000,
            cv_threshold=0.3,
            target_clusters=args.target_clusters,
            max_iterations=1 + args.n_extra_rounds,
            primer_prefix=args.primer_prefix,
            primer_suffix=args.primer_suffix,
            disable_merge=True,
            ref_length=args.ref_length,
            feddna_checkpoint=args.feddna_checkpoint,
            consensus_source='mv',
            fasta_source='mv_strict',
            zone_include_noise=True,
            rebirth_mode='nearest',

            gt_tags_file=args.gt_tags_file,
            gt_refs_file=args.gt_refs_file,
        )

        t_round = time.time()
        results = run_step2(step2_args)
        round_dur = time.time() - t_round

        if not results or 'next_round_files' not in results:
            print(f"❌ Round {round_idx}' 失败"); break

        nrf = results['next_round_files']
        current_labels    = nrf['labels']
        current_state     = nrf.get('state', current_state)
        current_consensus = nrf.get('consensus', current_consensus)
        history.append({
            'round': f"R{round_idx}' (frozen)",
            'labels': current_labels,
            'fasta': nrf.get('reference'),
            'duration_s': round_dur,
        })
        print(f"\n✅ Round {round_idx}' 完成 (耗时 {round_dur/60:.1f} min)")
        print(f"   labels: {current_labels}")
        print(f"   fasta:  {nrf.get('reference')}")

    # ── 5. 总结 ─────────────────────────────────────────────────────
    total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"📋 spike 完成 (总耗时 {total/60:.1f} min)")
    print("=" * 70)

    summary_path = os.path.join(args.spike_dir, 'spike_e_heavy_outputs.txt')
    with open(summary_path, 'w') as f:
        f.write("Spike E heavy: frozen R1 encoder, iterate R2'/R3'\n")
        f.write("=" * 60 + "\n\n")
        for h in history:
            f.write(f"\n{h['round']}\n")
            f.write(f"  labels: {h['labels']}\n")
            if h.get('fasta'):
                f.write(f"  fasta:  {h['fasta']}\n")
            if 'duration_s' in h:
                f.write(f"  time:   {h['duration_s']/60:.1f} min\n")
        f.write("\n\n下一步评估 (用 v19 同样的脚本):\n")
        for h in history[1:]:
            if h.get('fasta'):
                f.write(f"\n  # {h['round']}\n")
                f.write(f"  python eval_reconstruction.py \\\n")
                f.write(f"      --experiment_dir {os.path.dirname(os.path.dirname(h['fasta']))} \\\n")
                f.write(f"      --gt_tags_file <gt_tags> \\\n")
                f.write(f"      --gt_refs_file <gt_refs>\n")

    for h in history:
        line = f"  {h['round']:<25}"
        if h.get('fasta'):
            line += f"  fasta: ...{h['fasta'][-50:]}"
        print(line)
    print(f"\n💾 详细记录: {summary_path}\n")
    print("接下来: 用同 v19 一样的 eval_reconstruction.py 评估上面两个 fasta, "
          "把 SR 数字贴给 Eitan.")


if __name__ == '__main__':
    main()