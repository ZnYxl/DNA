#!/usr/bin/env python3
"""
gt_audit_probe.py — 合并诊断探针
=================================
放到 models/ 目录，在 step2_runner.py 中 import 后插入 4 行调用即可。

功能：
  1. 在 Step2 的 4 个关键阶段保存标签快照
  2. 自动对比相邻阶段的标签差异，分离出：
     - MNN 合并引入的污染
     - Zone III 隔离丢弃的 reads
     - 死数据复活引入的污染
  3. 对每对合并操作做 GT 正确/错误判定

用法（在 step2_runner.py 中插入）：
  见文件底部的 PATCH GUIDE 注释
"""
import os
import torch
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime


def _cluster_gt_majority(labels, gt_labels):
    """返回 {cluster_id: (majority_gt, purity, n_reads)}"""
    cluster_reads = defaultdict(list)
    for i in range(len(labels)):
        cid = int(labels[i])
        if cid < 0:
            continue
        gt = int(gt_labels[i])
        if gt >= 0:
            cluster_reads[cid].append(gt)

    result = {}
    for cid, gt_list in cluster_reads.items():
        counter = Counter(gt_list)
        maj_gt, maj_cnt = counter.most_common(1)[0]
        result[cid] = (maj_gt, maj_cnt / len(gt_list), len(gt_list))
    return result


class MergeAuditProbe:
    """
    在 step2_runner.py 开头创建一次，在 4 个阶段各调用一次 snapshot()。
    round 结束时调用 report() 输出诊断。

    用法:
        probe = MergeAuditProbe(gt_labels, save_dir)
        probe.snapshot("before_merge", labels)
        # ... merge ...
        probe.snapshot("after_merge", labels)
        # ... zone3 isolation ...
        probe.snapshot("after_zone3", labels)
        # ... revival ...
        probe.snapshot("after_revival", labels)
        probe.report(round_idx)
    """

    STAGES = ["before_merge", "after_merge", "after_zone3", "after_revival"]

    def __init__(self, gt_labels, save_dir=None):
        """
        Args:
            gt_labels: np.array(N,) 每条 read 的 GT cluster ID, -1 = 未匹配
            save_dir:  保存快照和报告的目录 (None = 不保存文件，只打印)
        """
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.numpy()
        self.gt_labels = gt_labels
        self.save_dir = save_dir
        self.snapshots = {}

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def snapshot(self, stage_name, labels):
        """保存某阶段的标签快照"""
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy().copy()
        else:
            labels = np.array(labels).copy()
        self.snapshots[stage_name] = labels

    def _find_merges_between(self, old_labels, new_labels):
        """找出从 old → new 的合并事件"""
        new_to_old = defaultdict(set)
        for i in range(len(old_labels)):
            oc = int(old_labels[i])
            nc = int(new_labels[i])
            if oc < 0 or nc < 0:
                continue
            new_to_old[nc].add(oc)

        merges = []
        for nc, oc_set in new_to_old.items():
            if len(oc_set) >= 2:
                merges.append((nc, sorted(oc_set)))
        return merges

    def _audit_merges(self, old_labels, new_labels, stage_name):
        """审计一个阶段的合并操作"""
        old_info = _cluster_gt_majority(old_labels, self.gt_labels)
        new_info = _cluster_gt_majority(new_labels, self.gt_labels)
        merges = self._find_merges_between(old_labels, new_labels)

        n_correct = 0
        n_wrong = 0
        total_contaminated = 0
        wrong_details = []

        for new_cid, old_cids in merges:
            gt_set = set()
            old_entries = []
            for oc in old_cids:
                if oc in old_info:
                    maj_gt, pur, n = old_info[oc]
                    gt_set.add(maj_gt)
                    old_entries.append((oc, maj_gt, pur, n))

            if len(gt_set) <= 1:
                n_correct += 1
            else:
                n_wrong += 1
                if new_cid in new_info:
                    _, new_pur, new_n = new_info[new_cid]
                    contaminated = int(new_n * (1 - new_pur))
                else:
                    contaminated = 0
                total_contaminated += contaminated
                wrong_details.append({
                    'new_cid': new_cid,
                    'old_clusters': old_entries,
                    'contaminated': contaminated,
                })

        return {
            'stage': stage_name,
            'n_merges': len(merges),
            'n_correct': n_correct,
            'n_wrong': n_wrong,
            'precision': n_correct / len(merges) if merges else 1.0,
            'contaminated': total_contaminated,
            'wrong_details': wrong_details,
        }

    def _count_label_changes(self, old_labels, new_labels, stage_name):
        """统计标签变化"""
        N = len(old_labels)
        n_to_noise = 0      # label >= 0 → -1
        n_revived = 0        # label == -1 → >= 0
        n_reassigned = 0     # label >= 0 → 不同的 >= 0
        n_unchanged = 0

        for i in range(N):
            old = int(old_labels[i])
            new = int(new_labels[i])
            if old == new:
                n_unchanged += 1
            elif old >= 0 and new < 0:
                n_to_noise += 1
            elif old < 0 and new >= 0:
                n_revived += 1
            elif old >= 0 and new >= 0 and old != new:
                n_reassigned += 1
            # old < 0 and new < 0: both noise, unchanged

        # 复活的 GT 审计
        revived_correct = 0
        revived_wrong = 0
        if n_revived > 0:
            new_info = _cluster_gt_majority(new_labels, self.gt_labels)
            for i in range(N):
                if int(old_labels[i]) < 0 and int(new_labels[i]) >= 0:
                    gt = int(self.gt_labels[i])
                    nc = int(new_labels[i])
                    if nc in new_info and gt >= 0:
                        maj_gt = new_info[nc][0]
                        if gt == maj_gt:
                            revived_correct += 1
                        else:
                            revived_wrong += 1

        return {
            'stage': stage_name,
            'to_noise': n_to_noise,
            'revived': n_revived,
            'revived_correct': revived_correct,
            'revived_wrong': revived_wrong,
            'reassigned': n_reassigned,
            'unchanged': n_unchanged,
        }

    def report(self, round_idx=0):
        """生成并打印完整诊断报告"""
        print(f"\n{'='*70}")
        print(f"  🔬 GT 合并诊断探针 — Round {round_idx}")
        print(f"{'='*70}")

        lines = []  # for file output

        # ── 阶段间对比 ──
        stage_pairs = [
            ("before_merge", "after_merge", "MNN 合并"),
            ("after_merge", "after_zone3", "Zone III 隔离"),
            ("after_zone3", "after_revival", "死数据复活"),
        ]

        for old_stage, new_stage, desc in stage_pairs:
            if old_stage not in self.snapshots or new_stage not in self.snapshots:
                print(f"  ⚠️ 缺少 {old_stage} 或 {new_stage} 快照，跳过 {desc}")
                continue

            old = self.snapshots[old_stage]
            new = self.snapshots[new_stage]

            print(f"\n  ── {desc} ({old_stage} → {new_stage}) ──")

            # 标签变化统计
            changes = self._count_label_changes(old, new, desc)
            print(f"     标签变 → 噪声(-1): {changes['to_noise']}")
            print(f"     复活(-1 → 簇):     {changes['revived']}")
            if changes['revived'] > 0:
                rev_prec = changes['revived_correct'] / (changes['revived_correct'] + changes['revived_wrong']) \
                    if (changes['revived_correct'] + changes['revived_wrong']) > 0 else 0
                print(f"       复活正确: {changes['revived_correct']}, "
                      f"复活错误: {changes['revived_wrong']}, "
                      f"精度: {rev_prec:.4f}")
            print(f"     重分配(簇→簇):     {changes['reassigned']}")

            # 合并审计（只有 MNN 阶段才有意义）
            if "merge" in desc.lower() or "合并" in desc:
                audit = self._audit_merges(old, new, desc)
                print(f"     ──── 合并精度 ────")
                print(f"     合并组数:    {audit['n_merges']}")
                print(f"     ✅ 正确:      {audit['n_correct']}")
                print(f"     ❌ 错误:      {audit['n_wrong']}")
                print(f"     🎯 精度:      {audit['precision']:.4f} ({audit['precision']*100:.1f}%)")
                print(f"     ☠️  污染reads: ~{audit['contaminated']}")

                # Top 5 错误详情
                if audit['wrong_details']:
                    audit['wrong_details'].sort(key=lambda x: -x['contaminated'])
                    n_show = min(5, len(audit['wrong_details']))
                    print(f"     Top {n_show} 错误:")
                    for d in audit['wrong_details'][:n_show]:
                        old_str = " + ".join(
                            f"C{oc}(GT{gt},n={n})"
                            for oc, gt, _, n in d['old_clusters'] if gt >= 0
                        )
                        print(f"       → NewC{d['new_cid']}: {old_str} "
                              f"(污染{d['contaminated']})")

                lines.append(f"[{desc}] merges={audit['n_merges']}, "
                            f"correct={audit['n_correct']}, wrong={audit['n_wrong']}, "
                            f"precision={audit['precision']:.4f}, "
                            f"contaminated={audit['contaminated']}")

        # ── 全局总结 ──
        if "before_merge" in self.snapshots and "after_revival" in self.snapshots:
            old_all = self.snapshots["before_merge"]
            new_all = self.snapshots["after_revival"]

            old_valid = int((old_all >= 0).sum())
            new_valid = int((new_all >= 0).sum())
            old_clusters = len(set(old_all[old_all >= 0].tolist()))
            new_clusters = len(set(new_all[new_all >= 0].tolist()))

            # 全阶段合并精度
            full_audit = self._audit_merges(old_all, new_all, "全阶段")
            print(f"\n  ── 全阶段总结 (before_merge → after_revival) ──")
            print(f"     簇数: {old_clusters} → {new_clusters}")
            print(f"     有效reads: {old_valid} → {new_valid} "
                  f"(丢失 {old_valid - new_valid})")
            print(f"     全阶段合并精度: {full_audit['precision']:.4f}")
            print(f"     全阶段污染reads: ~{full_audit['contaminated']}")

        # ── 保存文件 ──
        if self.save_dir:
            report_path = os.path.join(
                self.save_dir, f"merge_probe_round{round_idx}.txt"
            )
            with open(report_path, 'w') as f:
                f.write(f"Round {round_idx} Merge Audit Probe\n")
                f.write(f"Date: {datetime.now()}\n\n")
                for line in lines:
                    f.write(line + "\n")
            print(f"\n  💾 探针报告: {report_path}")

        print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# PATCH GUIDE: 如何在 step2_runner.py 中插入探针
# ═══════════════════════════════════════════════════════════════════
#
# 总共需要插入 7 行代码，分布在 4 个位置。
#
# ── 位置 0: 文件顶部 import 区 ──
#   加一行:
#     from models.gt_audit_probe import MergeAuditProbe
#
#
# ── 位置 1: 质心计算之后、MNN 合并之前 ──
#   找到这段代码:
#     centroids, cluster_sizes = compute_centroids_weighted(...)
#     delta = compute_global_delta(...)
#
#   在 delta 计算之后、merge_close_centroids 调用之前，插入:
#
#     # ★ GT 合并探针：初始化 + 合并前快照
#     _gt_np = gt_labels.numpy() if isinstance(gt_labels, torch.Tensor) else gt_labels
#     _probe = MergeAuditProbe(_gt_np, save_dir=os.path.join(args.experiment_dir, 'results', f'iter_{round_idx}_step2'))
#     _probe.snapshot("before_merge", labels_tensor)
#
#
# ── 位置 2: MNN 合并之后、Zone III 隔离之前 ──
#   找到这段代码:
#     centroids, labels_tensor, merge_stats, cluster_sizes = merge_close_centroids(...)
#     new_labels = labels_tensor.clone()
#     ...Zone III 标签隔离...
#
#   在 merge_close_centroids 返回之后、Zone III 隔离之前，插入:
#
#     _probe.snapshot("after_merge", labels_tensor)
#
#
# ── 位置 3: Zone III 隔离之后、死数据复活之前 ──
#   找到:
#     new_labels[zone_ids == 3] = -1
#     print(f"   🔒 Zone III ...")
#
#   在 print 之后、死数据复活代码之前，插入:
#
#     _probe.snapshot("after_zone3", new_labels)
#
#
# ── 位置 4: 死数据复活之后、GT 评估之前 ──
#   找到死数据复活的结尾（通常是 "✨ 成功复活: ..." 的 print 附近）
#   在复活逻辑全部结束之后，插入:
#
#     _probe.snapshot("after_revival", new_labels)
#     _probe.report(round_idx)
#
# ═══════════════════════════════════════════════════════════════════