#!/usr/bin/env python3
"""
SSI-EC 实验日志解析器
用法：python parse_log.py exp1_ours_full_log.txt
输出：终端摘要 + parsed_report.txt
"""

import sys
import re
import os
from collections import defaultdict

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "exp1_ours_full_log.txt"

# ─────────────────────────────────────────────────────────────────────────────
# 正则表达式（对应训练代码的每一种打印格式）
# ─────────────────────────────────────────────────────────────────────────────
RE_ROUND    = re.compile(r'Round\s+(\d+)\s*/\s*(\d+)')
RE_EPOCH_OK = re.compile(
    r'✅ Epoch\s+(\d+).*?Loss:\s*([\d.]+).*?Str:\s*([\d.]+).*?Recon:\s*([\d.]+).*?U_epi:\s*([\d.]+)'
)
RE_PROBE_A  = re.compile(
    r'探针 A.*?w\(干净-干净\):\s*([\d.nan]+).*?w\(含脏-任意\):\s*([\d.nan]+).*?比值:\s*([\d.nan]+)x'
)
RE_PROBE_B  = re.compile(
    r'探针 B.*?cos_pos:\s*([\d.nan]+).*?cos_neg:\s*([\d.nan]+).*?margin:\s*([\d.nan-]+)'
)
RE_ZONE     = re.compile(
    r'Zone\s+(I{1,3}|III?)\s*\([^)]+\):\s*([\d,]+)\s*\(\s*([\d.]+)%\)'
)
RE_GT_PURITY = re.compile(
    r'Cluster Purity[：:]\s*([\d.]+)'
)
RE_GT_PERFECT = re.compile(
    r'Perfect Cluster Rate[：:]\s*(\d+)/(\d+)\s*\(([\d.]+)\)'
)
RE_CONSENSUS = re.compile(r'生成\s+(\d+)\s+个.*?consensus|consensus.*?(\d+)\s+个簇')
RE_LABEL_CHG = re.compile(r'标签变化率.*?([\d.]+)%')
RE_BATCH_SZ  = re.compile(r'Batches:\s*(\d+)\s*\(avg\s*([\d.]+)')
RE_CKPT      = re.compile(r'Checkpoint 保存:\s*(.+)')
RE_ERROR     = re.compile(r'(Error|error|Traceback|❌|RuntimeError|AssertionError|CUDA out of memory)')
RE_DELTA     = re.compile(r'Global Delta\s*=\s*([\d.]+)')
RE_REFINE    = re.compile(
    r'Zone I 保持:\s*([\d,]+).*?Zone II 重分配:\s*([\d,]+).*?Zone II → 噪声:\s*([\d,]+).*?Zone III 丢弃:\s*([\d,]+)',
    re.DOTALL
)

# ─────────────────────────────────────────────────────────────────────────────
# 解析
# ─────────────────────────────────────────────────────────────────────────────
rounds = defaultdict(lambda: {
    'epochs': [],       # [{'epoch':1, 'loss':..., 'str':..., 'recon':..., 'u_epi':...,
                        #   'probe_a_cc':..., 'probe_a_da':..., 'probe_ratio':...,
                        #   'cos_pos':..., 'cos_neg':..., 'margin':...}]
    'zones': [],        # [{'zone1':N, 'zone2':N, 'zone3':N}]  per Step2 run
    'gt_purity': [],
    'gt_perfect': [],
    'global_delta': [],
    'refine': [],       # [{'z1_kept':N, 'z2_reassign':N, 'z2_noise':N, 'z3_dirty':N}]
    'label_change': [],
    'checkpoints': [],
    'errors': [],
})

current_round = 0
lines = []
try:
    with open(LOG_FILE, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"❌ 文件不存在: {LOG_FILE}")
    sys.exit(1)

print(f"📂 读取日志: {LOG_FILE}  ({len(lines)} 行)")

# 多行合并：refine_reads的输出跨多行，先把相邻行拼成块
text_blocks = []
buf = ""
for line in lines:
    buf += line
    if len(buf) > 2000:
        text_blocks.append(buf)
        buf = ""
if buf:
    text_blocks.append(buf)

full_text = "".join(lines)

# 逐行解析
for i, line in enumerate(lines):
    line = line.rstrip()

    # 当前 Round
    m = RE_ROUND.search(line)
    if m:
        current_round = int(m.group(1))

    # Epoch 结束行
    m = RE_EPOCH_OK.search(line)
    if m:
        rounds[current_round]['epochs'].append({
            'epoch':  int(m.group(1)),
            'loss':   float(m.group(2)),
            'str':    float(m.group(3)),
            'recon':  float(m.group(4)),
            'u_epi':  float(m.group(5)),
        })

    # 探针 A
    m = RE_PROBE_A.search(line)
    if m and rounds[current_round]['epochs']:
        def _f(s):
            try: return float(s)
            except: return float('nan')
        ep = rounds[current_round]['epochs'][-1]
        ep['probe_a_cc']    = _f(m.group(1))
        ep['probe_a_da']    = _f(m.group(2))
        ep['probe_ratio']   = _f(m.group(3))

    # 探针 B
    m = RE_PROBE_B.search(line)
    if m and rounds[current_round]['epochs']:
        def _f(s):
            try: return float(s)
            except: return float('nan')
        ep = rounds[current_round]['epochs'][-1]
        ep['cos_pos'] = _f(m.group(1))
        ep['cos_neg'] = _f(m.group(2))
        ep['margin']  = _f(m.group(3))

    # Zone 分布
    m = RE_ZONE.search(line)
    if m:
        zone_name = m.group(1).strip()
        zone_cnt  = int(m.group(2).replace(',', ''))
        zone_pct  = float(m.group(3))
        rz = rounds[current_round]['zones']
        if not rz or ('zone3' in rz[-1]):
            rz.append({'zone1': 0, 'zone2': 0, 'zone3': 0})
        cur = rz[-1]
        if zone_name in ('I',):
            cur['zone1'] = zone_cnt; cur['pct1'] = zone_pct
        elif zone_name in ('II',):
            cur['zone2'] = zone_cnt; cur['pct2'] = zone_pct
        elif zone_name in ('III',):
            cur['zone3'] = zone_cnt; cur['pct3'] = zone_pct

    # GT 评估
    m = RE_GT_PURITY.search(line)
    if m:
        rounds[current_round]['gt_purity'].append(float(m.group(1)))

    m = RE_GT_PERFECT.search(line)
    if m:
        rounds[current_round]['gt_perfect'].append({
            'perfect': int(m.group(1)),
            'total':   int(m.group(2)),
            'rate':    float(m.group(3)),
        })

    # Global Delta
    m = RE_DELTA.search(line)
    if m:
        rounds[current_round]['global_delta'].append(float(m.group(1)))

    # 标签变化率
    m = RE_LABEL_CHG.search(line)
    if m:
        rounds[current_round]['label_change'].append(float(m.group(1)))

    # Checkpoint 路径
    m = RE_CKPT.search(line)
    if m:
        rounds[current_round]['checkpoints'].append(m.group(1).strip())

    # 错误
    m = RE_ERROR.search(line)
    if m:
        rounds[current_round]['errors'].append(f"L{i+1}: {line.strip()[:120]}")

# refine 统计（跨多行，用全文搜索）
for block in text_blocks:
    m = RE_REFINE.search(block)
    if m:
        def _i(s): return int(s.replace(',', ''))
        # 找到这个block时的round（粗略用current_round，多轮会覆盖）
        rm = RE_ROUND.search(block)
        rnd = int(rm.group(1)) if rm else current_round
        rounds[rnd]['refine'].append({
            'z1_kept':     _i(m.group(1)),
            'z2_reassign': _i(m.group(2)),
            'z2_noise':    _i(m.group(3)),
            'z3_dirty':    _i(m.group(4)),
        })

# ─────────────────────────────────────────────────────────────────────────────
# 输出报告
# ─────────────────────────────────────────────────────────────────────────────
def _nan(v, fmt='.4f'):
    import math
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'nan'
    return format(v, fmt)

lines_out = []
def pr(s=""):
    print(s)
    lines_out.append(s)

pr("=" * 70)
pr("  SSI-EC 实验日志解析报告")
pr(f"  来源: {LOG_FILE}")
pr("=" * 70)

for rnd in sorted(rounds.keys()):
    if rnd == 0 and not rounds[0]['epochs']:
        continue
    data = rounds[rnd]
    pr(f"\n{'─'*70}")
    pr(f"  Round {rnd}")
    pr(f"{'─'*70}")

    # ── Step 1 训练 ──────────────────────────────────────────────────────────
    epochs = data['epochs']
    if epochs:
        pr(f"\n【Step 1 训练】共 {len(epochs)} 个 epoch")
        pr(f"  {'Ep':>3}  {'Loss':>8}  {'Str':>7}  {'Recon':>7}  {'U_epi':>7}"
           f"  {'w_cc':>6}  {'w_da':>6}  {'ratio':>6}  {'cos+':>6}  {'cos-':>6}  {'margin':>7}")
        for ep in epochs:
            pr(f"  {ep['epoch']:>3}  {ep['loss']:>8.4f}  {ep['str']:>7.1f}"
               f"  {ep['recon']:>7.4f}  {ep['u_epi']:>7.4f}"
               f"  {_nan(ep.get('probe_a_cc')):>6}"
               f"  {_nan(ep.get('probe_a_da')):>6}"
               f"  {_nan(ep.get('probe_ratio')):>6}"
               f"  {_nan(ep.get('cos_pos')):>6}"
               f"  {_nan(ep.get('cos_neg')):>6}"
               f"  {_nan(ep.get('margin')):>7}")

        # Loss 下降分析
        if len(epochs) >= 2:
            loss_drop = epochs[0]['loss'] - epochs[-1]['loss']
            recon_drop = epochs[0]['recon'] - epochs[-1]['recon']
            pr(f"\n  Loss 下降: {epochs[0]['loss']:.4f} → {epochs[-1]['loss']:.4f}"
               f"  (↓{loss_drop:.4f})")
            pr(f"  Recon 下降: {epochs[0]['recon']:.4f} → {epochs[-1]['recon']:.4f}"
               f"  (↓{recon_drop:.4f})")

        # cos_pos 趋势
        cos_vals = [ep.get('cos_pos') for ep in epochs if ep.get('cos_pos') is not None
                    and not (isinstance(ep.get('cos_pos'), float) and ep.get('cos_pos') != ep.get('cos_pos'))]
        if len(cos_vals) >= 2:
            pr(f"  cos_pos 趋势: {cos_vals[0]:.4f} → {cos_vals[-1]:.4f}"
               f"  ({'↑上升' if cos_vals[-1] > cos_vals[0] else '↓下降'})")

        # 探针 A 比值分析
        ratios = [ep.get('probe_ratio') for ep in epochs
                  if ep.get('probe_ratio') is not None
                  and not (isinstance(ep.get('probe_ratio'), float)
                           and ep.get('probe_ratio') != ep.get('probe_ratio'))]
        if ratios:
            pr(f"  探针A比值: 最大={max(ratios):.1f}x  最终={ratios[-1]:.1f}x"
               f"  {'✅ 两极分化有效' if max(ratios) > 5 else '⚠️ 两极分化不足(<5x)'}")
        else:
            pr(f"  探针A比值: 全为 nan（U_ale 未分化，Round 1 正常现象）")

    # ── Step 2 ───────────────────────────────────────────────────────────────
    pr(f"\n【Step 2 推理与修正】")

    zones = data['zones']
    if zones:
        for j, z in enumerate(zones):
            pr(f"  三区制 #{j+1}:")
            pr(f"    Zone I  (净): {z.get('zone1',0):>10,}  ({z.get('pct1',0):5.1f}%)")
            pr(f"    Zone II (难): {z.get('zone2',0):>10,}  ({z.get('pct2',0):5.1f}%)")
            pr(f"    Zone III(脏): {z.get('zone3',0):>10,}  ({z.get('pct3',0):5.1f}%)")
            # 诊断
            pct3 = z.get('pct3', 0)
            if pct3 > 20:
                pr(f"    ⚠️  Zone III 占比过高({pct3:.1f}%) → 检查是否有 EMA 或 U_ale 污染")
            elif pct3 < 5:
                pr(f"    ⚠️  Zone III 占比过低({pct3:.1f}%) → U_ale 分布未分化，噪声未被正确识别")
            else:
                pr(f"    ✅ Zone III 占比正常")

    if data['global_delta']:
        for d in data['global_delta']:
            pr(f"  Global Delta: {d:.4f}")

    if data['refine']:
        for r in data['refine']:
            total = r['z1_kept'] + r['z2_reassign'] + r['z2_noise'] + r['z3_dirty']
            assigned = r['z1_kept'] + r['z2_reassign']
            pr(f"  标签修正:")
            pr(f"    Zone I 保持:       {r['z1_kept']:>10,}")
            pr(f"    Zone II 重分配:    {r['z2_reassign']:>10,}")
            pr(f"    Zone II → 噪声:    {r['z2_noise']:>10,}")
            pr(f"    Zone III 丢弃:     {r['z3_dirty']:>10,}")
            if total > 0:
                pr(f"    总分配率: {assigned/total*100:.1f}%  噪声率: {(r['z2_noise']+r['z3_dirty'])/total*100:.1f}%")

    if data['gt_purity']:
        for p in data['gt_purity']:
            status = '✅' if p > 0.974 else ('⚠️' if p > 0.92 else '❌')
            pr(f"  GT Cluster Purity: {p:.4f}  {status}  (Clover基线=0.974)")

    if data['gt_perfect']:
        for gp in data['gt_perfect']:
            pr(f"  GT Perfect Cluster Rate: {gp['perfect']}/{gp['total']} = {gp['rate']:.4f}"
               f"  {'✅' if gp['rate'] > 0.20 else '⚠️'}")

    if data['label_change']:
        for lc in data['label_change']:
            pr(f"  标签变化率: {lc:.2f}%  {'✅ 收敛中' if lc < 5 else '⚠️ 变化仍较大'}")

    if data['checkpoints']:
        pr(f"  Checkpoint: {data['checkpoints'][-1]}")

    # ── 错误 ─────────────────────────────────────────────────────────────────
    if data['errors']:
        pr(f"\n【⚠️  发现错误/警告】共 {len(data['errors'])} 处:")
        for e in data['errors'][:20]:   # 最多显示20条
            pr(f"  {e}")
        if len(data['errors']) > 20:
            pr(f"  ... 还有 {len(data['errors'])-20} 条")

# ─────────────────────────────────────────────────────────────────────────────
# 全局总结
# ─────────────────────────────────────────────────────────────────────────────
pr(f"\n{'=' * 70}")
pr("  全局总结")
pr(f"{'=' * 70}")

all_purity = []
all_perfect = []
for rnd in sorted(rounds.keys()):
    all_purity.extend(rounds[rnd]['gt_purity'])
    all_perfect.extend([gp['rate'] for gp in rounds[rnd]['gt_perfect']])

if all_purity:
    pr(f"\n  Cluster Purity  各轮: {' → '.join(f'{p:.4f}' for p in all_purity)}")
    trend = '↑改善' if len(all_purity) >= 2 and all_purity[-1] > all_purity[0] else '↓退化' if len(all_purity) >= 2 else '-'
    pr(f"                  趋势: {trend}")

if all_perfect:
    pr(f"  Perfect Rate    各轮: {' → '.join(f'{p:.4f}' for p in all_perfect)}")
    trend = '↑改善' if len(all_perfect) >= 2 and all_perfect[-1] > all_perfect[0] else '↓退化' if len(all_perfect) >= 2 else '-'
    pr(f"                  趋势: {trend}")

# Nan 统计（帮助定位探针 bug）
nan_rounds = []
for rnd in sorted(rounds.keys()):
    has_nan = any(
        isinstance(ep.get('probe_ratio'), float) and ep.get('probe_ratio') != ep.get('probe_ratio')
        for ep in rounds[rnd]['epochs']
    )
    if has_nan:
        nan_rounds.append(rnd)
if nan_rounds:
    pr(f"\n  ⚠️  以下 Round 存在探针 nan（U_ale 未分化或探针 bug）: {nan_rounds}")
    pr(f"     → 建议修复 step1_train.py 中的探针 nan 检查逻辑（各探针独立计数）")

all_errors = sum(len(rounds[r]['errors']) for r in rounds)
if all_errors:
    pr(f"\n  ⚠️  全日志共发现 {all_errors} 处错误/警告，请检查各 Round 的错误部分")
else:
    pr(f"\n  ✅ 全日志未发现报错")

pr("")

# 保存到文件
out_path = LOG_FILE.replace('.txt', '_parsed.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines_out))
pr(f"💾 报告已保存至: {out_path}")