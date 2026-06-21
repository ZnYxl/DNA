#!/usr/bin/env python3
"""
parse_log.py — SSI-EC 实验日志提取脚本

用法:
    python parse_log.py exp1_ours_v2.log
    python parse_log.py exp1_ours_v2.log --output report.txt
"""
import re
import sys
import argparse
from collections import defaultdict


def parse_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ── 数据结构 ──────────────────────────────────────────────
    rounds = {}          # round_idx → {...}
    current_round = None
    current_step  = None   # 'step1' or 'step2'

    for i, line in enumerate(lines):
        line = line.rstrip()

        # ── Round 检测 ────────────────────────────────────────
        m = re.search(r'Round\s+(\d+)\s*/\s*(\d+)', line)
        if m and '🔄' in line:
            current_round = int(m.group(1))
            total_rounds  = int(m.group(2))
            if current_round not in rounds:
                rounds[current_round] = {
                    'total_rounds': total_rounds,
                    'step1': {'epochs': []},
                    'step2': {},
                }
            current_step = None
            continue

        if current_round is None:
            continue

        # ── Step 检测 ─────────────────────────────────────────
        if '[Step 1]' in line:
            current_step = 'step1'
        elif '[Step 2]' in line:
            current_step = 'step2'

        # ── Step 1: Epoch 指标 ───────────────────────────────
        if current_step == 'step1':
            # ✅ Epoch N (Xs) | Loss: X | Str: X | Recon: X | U_epi: X
            m = re.search(
                r'Epoch\s+(\d+).*?Loss:\s*([\d.]+).*?Str:\s*([\d.]+)'
                r'.*?Recon:\s*([\d.]+).*?U_epi:\s*([\d.]+)', line
            )
            if m and '✅' in line:
                # 提取耗时
                tm = re.search(r'\(([\d.]+)s\)', line)
                epoch_data = {
                    'epoch':   int(m.group(1)),
                    'loss':    float(m.group(2)),
                    'str':     float(m.group(3)),
                    'recon':   float(m.group(4)),
                    'u_epi':   float(m.group(5)),
                    'time_s':  float(tm.group(1)) if tm else None,
                }
                rounds[current_round]['step1']['epochs'].append(epoch_data)

            # 探针 A
            m = re.search(r'w\(干净-干净\):\s*([\d.]+).*?w\(含脏-任意\):\s*([\d.]+).*?比值:\s*([\d.]+)x', line)
            if m and rounds[current_round]['step1']['epochs']:
                rounds[current_round]['step1']['epochs'][-1].update({
                    'w_cc': float(m.group(1)),
                    'w_da': float(m.group(2)),
                    'ratio': float(m.group(3)),
                })

            # 探针 B
            m = re.search(r'cos_pos:\s*([\d.]+).*?cos_neg:\s*([\d.]+).*?margin:\s*([\d.]+)', line)
            if m and rounds[current_round]['step1']['epochs']:
                rounds[current_round]['step1']['epochs'][-1].update({
                    'cos_pos': float(m.group(1)),
                    'cos_neg': float(m.group(2)),
                    'margin':  float(m.group(3)),
                })

        # ── Step 2: 三区制 ────────────────────────────────────
        if current_step == 'step2':
            s2 = rounds[current_round]['step2']

            # Zone I/II/III
            m = re.search(r'Zone I\s*\(Safe\):\s*([\d,]+)\s*\(\s*([\d.]+)%\)', line)
            if m:
                s2['zone1_n']   = int(m.group(1).replace(',', ''))
                s2['zone1_pct'] = float(m.group(2))

            m = re.search(r'Zone II\s*\(Hard\):\s*([\d,]+)\s*\(\s*([\d.]+)%\)', line)
            if m:
                s2['zone2_n']   = int(m.group(1).replace(',', ''))
                s2['zone2_pct'] = float(m.group(2))

            m = re.search(r'Zone III\s*\(Dirty\):\s*([\d,]+)\s*\(\s*([\d.]+)%\)', line)
            if m:
                s2['zone3_n']   = int(m.group(1).replace(',', ''))
                s2['zone3_pct'] = float(m.group(2))

            # MNN 合并总结
            m = re.search(r'簇数:\s*([\d,]+)\s*→\s*([\d,]+)', line)
            if m and '合并总结' not in lines[max(0,i-5):i+1][0]:
                s2['clusters_before'] = int(m.group(1).replace(',', ''))
                s2['clusters_after']  = int(m.group(2).replace(',', ''))

            m = re.search(r'合并次数:\s*(\d+)', line)
            if m:
                s2['mnn_merges'] = int(m.group(1))

            m = re.search(r'耗时:\s*([\d.]+)s', line)
            if m and 'mnn_time_s' not in s2:
                s2['mnn_time_s'] = float(m.group(1))

            m = re.search(r'簇大小: max=(\d+), median=(\d+), min=(\d+)', line)
            if m:
                s2['size_max']    = int(m.group(1))
                s2['size_median'] = int(m.group(2))
                s2['size_min']    = int(m.group(3))

            # Global Delta
            m = re.search(r'Global Delta\s*=\s*([\d.]+)', line)
            if m:
                s2['global_delta'] = float(m.group(1))

            # GT 评估
            m = re.search(r'Cluster Purity:\s*([\d.]+)', line)
            if m:
                s2['purity'] = float(m.group(1))

            m = re.search(r'Perfect Cluster Rate:\s*(\d+)/(\d+)\s*\(([\d.]+)\)', line)
            if m:
                s2['perfect_n']     = int(m.group(1))
                s2['total_clusters']= int(m.group(2))
                s2['perfect_rate']  = float(m.group(3))

            # CV 困难度
            m = re.search(r'困难簇\(≥([\d.]+)\)=(\d+).*?完美簇=(\d+).*?中位CV=([\d.]+)', line)
            if m:
                s2['cv_threshold']   = float(m.group(1))
                s2['hard_clusters']  = int(m.group(2))
                s2['easy_clusters']  = int(m.group(3))
                s2['median_cv']      = float(m.group(4))

        # ── 标签变化率 ────────────────────────────────────────
        m = re.search(r'标签变化率:\s*([\d.]+)\s*\(([\d.]+)%\)', line)
        if m and current_round:
            rounds[current_round]['label_change_rate'] = float(m.group(1))

    return rounds


def format_report(rounds):
    lines = []
    sep  = '=' * 70
    sep2 = '-' * 70

    lines.append(sep)
    lines.append('  SSI-EC 实验日志报告')
    lines.append(sep)

    for rnd in sorted(rounds.keys()):
        r = rounds[rnd]
        lines.append(f'\n{"="*70}')
        lines.append(f'  Round {rnd} / {r["total_rounds"]}')
        lines.append(f'{"="*70}')

        # ── Step 1 ────────────────────────────────────────────
        lines.append('\n【Step 1 训练】')
        epochs = r['step1']['epochs']
        if epochs:
            header = f"{'Ep':>4}  {'Loss':>8}  {'Str':>7}  {'Recon':>7}  {'U_epi':>7}  {'w_cc':>7}  {'w_da':>7}  {'ratio':>6}  {'cos+':>7}  {'cos-':>7}  {'margin':>7}"
            lines.append(header)
            lines.append(sep2)
            for e in epochs:
                w_cc  = f"{e.get('w_cc',float('nan')):7.4f}" if 'w_cc' in e else '    N/A'
                w_da  = f"{e.get('w_da',float('nan')):7.4f}" if 'w_da' in e else '    N/A'
                ratio = f"{e.get('ratio',float('nan')):6.1f}x" if 'ratio' in e else '   N/Ax'
                cp    = f"{e.get('cos_pos',float('nan')):7.4f}" if 'cos_pos' in e else '    N/A'
                cn    = f"{e.get('cos_neg',float('nan')):7.4f}" if 'cos_neg' in e else '    N/A'
                mg    = f"{e.get('margin',float('nan')):7.4f}" if 'margin' in e else '    N/A'
                lines.append(
                    f"{e['epoch']:>4}  {e['loss']:>8.4f}  {e['str']:>7.1f}  "
                    f"{e['recon']:>7.4f}  {e['u_epi']:>7.4f}  "
                    f"{w_cc}  {w_da}  {ratio}  {cp}  {cn}  {mg}"
                )

            first, last = epochs[0], epochs[-1]
            lines.append(sep2)
            lines.append(f"  Loss  下降: {first['loss']:.4f} → {last['loss']:.4f}  (↓{first['loss']-last['loss']:.4f})")
            lines.append(f"  Recon 下降: {first['recon']:.4f} → {last['recon']:.4f}  (↓{first['recon']-last['recon']:.4f})")
            lines.append(f"  Str   增长: {first['str']:.1f} → {last['str']:.1f}")
            if 'margin' in last:
                trend = '↑上升' if last['margin'] > first.get('margin', 0) else '↓下降'
                lines.append(f"  cos_pos 趋势: {first.get('cos_pos','N/A')} → {last['cos_pos']:.4f}  ({trend})")
            if 'ratio' in last:
                max_ratio = max(e.get('ratio', 0) for e in epochs)
                flag = '⚠️ 两极分化不足(<5x)' if max_ratio < 5 else '✅ 两极分化良好'
                lines.append(f"  探针A比值: 最大={max_ratio:.1f}x  最终={last['ratio']:.1f}x  {flag}")
        else:
            lines.append('  （未找到 epoch 数据）')

        # ── Step 2 ────────────────────────────────────────────
        lines.append('\n【Step 2 聚类修正】')
        s2 = r['step2']
        if s2:
            # 三区制
            if 'zone1_pct' in s2:
                lines.append(f"  三区制划分:")
                lines.append(f"    Zone I  (Safe):  {s2.get('zone1_n',0):>10,}  ({s2.get('zone1_pct',0):5.1f}%)")
                lines.append(f"    Zone II (Hard):  {s2.get('zone2_n',0):>10,}  ({s2.get('zone2_pct',0):5.1f}%)")
                lines.append(f"    Zone III(Dirty): {s2.get('zone3_n',0):>10,}  ({s2.get('zone3_pct',0):5.1f}%)")

            # MNN 合并
            if 'clusters_before' in s2:
                lines.append(f"\n  MNN 安全合并:")
                lines.append(f"    簇数: {s2['clusters_before']:,} → {s2['clusters_after']:,}  (减少 {s2['clusters_before']-s2['clusters_after']:,})")
                lines.append(f"    合并次数: {s2.get('mnn_merges', 'N/A')}")
                lines.append(f"    耗时: {s2.get('mnn_time_s', 'N/A')}s")
                if 'size_max' in s2:
                    lines.append(f"    簇大小: max={s2['size_max']}, median={s2['size_median']}, min={s2['size_min']}")

            if 'global_delta' in s2:
                lines.append(f"\n  Global Delta: {s2['global_delta']:.4f}")

            # GT 评估
            if 'purity' in s2:
                lines.append(f"\n  GT 评估:")
                lines.append(f"    Cluster Purity:       {s2['purity']:.4f}")
                lines.append(f"    Perfect Cluster Rate: {s2.get('perfect_n','N/A')}/{s2.get('total_clusters','N/A')}  ({s2.get('perfect_rate',0):.4f})")

            # CV 困难度
            if 'median_cv' in s2:
                lines.append(f"\n  簇困难度 (CV):")
                lines.append(f"    中位 CV:   {s2['median_cv']:.4f}")
                lines.append(f"    阈值:      {s2.get('cv_threshold', 0.3)}")
                lines.append(f"    困难簇:    {s2.get('hard_clusters', 0)}")
                lines.append(f"    完美簇:    {s2.get('easy_clusters', 0)}")
                if s2.get('hard_clusters', 0) == 0:
                    lines.append(f"    ⚠️  困难簇=0，建议将 cv_threshold 降低至中位CV的2-3倍 ≈ {s2['median_cv']*2:.3f}")
        else:
            lines.append('  （未找到 Step 2 数据）')

        # ── 标签变化率 ────────────────────────────────────────
        if 'label_change_rate' in r:
            cr = r['label_change_rate']
            bar = '█' * min(40, int(cr * 100)) + '░' * max(0, 40 - int(cr * 100))
            lines.append(f"\n  标签变化率: {cr:.4f} ({cr*100:.2f}%)  {bar}")
        else:
            lines.append('\n  标签变化率: N/A（首轮）')

    # ── 跨轮对比 ──────────────────────────────────────────────
    lines.append(f'\n{sep}')
    lines.append('  跨轮对比总览')
    lines.append(sep)

    # Step 1 最终 epoch 对比
    lines.append('\n  Step 1 最终 epoch:')
    lines.append(f"  {'Round':>6}  {'Loss':>8}  {'Str':>7}  {'Recon':>7}  {'Margin':>7}  {'比值':>7}  {'Purity':>8}  {'PCR':>8}")
    lines.append(f"  {'-'*66}")
    for rnd in sorted(rounds.keys()):
        r = rounds[rnd]
        epochs = r['step1']['epochs']
        s2     = r['step2']
        if not epochs:
            continue
        last   = epochs[-1]
        purity = f"{s2.get('purity', float('nan')):.4f}" if 'purity' in s2 else '   N/A  '
        pcr    = f"{s2.get('perfect_rate', float('nan')):.4f}" if 'perfect_rate' in s2 else '   N/A  '
        margin = f"{last.get('margin', float('nan')):.4f}" if 'margin' in last else '   N/A '
        ratio  = f"{last.get('ratio', float('nan')):.1f}x" if 'ratio' in last else '  N/Ax'
        lines.append(
            f"  {rnd:>6}  {last['loss']:>8.4f}  {last['str']:>7.1f}  "
            f"{last['recon']:>7.4f}  {margin:>7}  {ratio:>7}  {purity:>8}  {pcr:>8}"
        )

    # 合并收敛趋势
    lines.append('\n  MNN 合并收敛趋势:')
    for rnd in sorted(rounds.keys()):
        s2 = rounds[rnd]['step2']
        if 'clusters_before' in s2:
            lines.append(f"    Round {rnd}: {s2['clusters_before']:>6,} → {s2['clusters_after']:>6,}  (减少 {s2['clusters_before']-s2['clusters_after']:>5,})")

    lines.append(f'\n{sep}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='SSI-EC 日志提取')
    parser.add_argument('log_path', type=str, help='日志文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径（默认打印到终端）')
    args = parser.parse_args()

    rounds = parse_log(args.log_path)
    report = format_report(rounds)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'✅ 报告已保存: {args.output}')
    else:
        print(report)


if __name__ == '__main__':
    main()