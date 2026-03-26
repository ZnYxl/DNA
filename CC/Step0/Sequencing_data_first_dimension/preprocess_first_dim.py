#!/usr/bin/env python3
"""
Sequencing_data_first_dimension 预处理 → Clover 输入（tag 模式）
处理逻辑（对齐师姐流程）：
  1. 去 N
  2. 长度过滤 [ref_len ± margin]
  3. 过滤后 reads < min_reads 的簇整体丢弃

输入: output.txt (BWA比对后 tag<TAB>read)
输出: clover_input.txt (tag<SPACE>read, Clover tag 模式)

Clover 运行命令:
  cd /mnt/st_data/liangxinyi/code/CC/Step0/Clover
  python -m clover.main -I <path>/clover_input.txt -L 196 -T 11826
"""
from collections import defaultdict

def preprocess(input_file, output_file, ref_len=196, margin=5, min_reads_per_cluster=5):
    min_len = ref_len - margin
    max_len = ref_len + margin

    # ── Pass 1: 读入 + 过滤 + 按 tag 分组 ──
    clusters = defaultdict(list)
    total_lines = 0
    n_dropped = 0
    len_dropped = 0
    format_dropped = 0

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            parts = line.split('\t', 1)
            if len(parts) != 2:
                format_dropped += 1
                continue
            tag, seq = parts

            if 'N' in seq.upper():
                n_dropped += 1
                continue

            if not (min_len <= len(seq) <= max_len):
                len_dropped += 1
                continue

            clusters[tag].append(seq)

    # ── Pass 2: 丢弃 reads 不足的簇 + 写入 ──
    total_tags_before = len(clusters)
    small_tags = {t for t, seqs in clusters.items() if len(seqs) < min_reads_per_cluster}
    valid_tags = set(clusters.keys()) - small_tags
    dropped_reads_from_small = sum(len(clusters[t]) for t in small_tags)

    valid_reads = 0
    with open(output_file, 'w') as f:
        for tag in sorted(valid_tags, key=lambda x: int(x)):
            for seq in clusters[tag]:
                f.write(f"{tag} {seq}\n")  # 空格分隔，Clover tag 模式
                valid_reads += 1

    # ── 统计报告 ──
    print("=" * 55)
    print("  Seq_1D 预处理报告")
    print("=" * 55)
    print(f"  输入文件:          {input_file}")
    print(f"  参考序列长度:      {ref_len}bp, 允许范围 [{min_len}, {max_len}]")
    print(f"  最少 reads/簇:     {min_reads_per_cluster}")
    print(f"  ─────────────────────────────────────")
    print(f"  总行数:            {total_lines:,}")
    print(f"  格式错误剔除:      {format_dropped:,}")
    print(f"  因含 N 剔除:       {n_dropped:,}")
    print(f"  因长度不合格剔除:  {len_dropped:,}")
    print(f"  ─────────────────────────────────────")
    print(f"  过滤后 tag 数:     {total_tags_before:,}")
    print(f"  因 reads<{min_reads_per_cluster} 丢弃:   "
          f"{len(small_tags):,} 个 tag ({dropped_reads_from_small:,} 条 reads)")
    print(f"  ─────────────────────────────────────")
    print(f"  最终有效 tag 数:   {len(valid_tags):,}")
    print(f"  最终有效 reads:    {valid_reads:,}")
    print(f"  输出文件:          {output_file}")
    print("=" * 55)
    print()
    print("  下一步: 跑 Clover")
    print("  cd /mnt/st_data/liangxinyi/code/CC/Step0/Clover")
    print(f"  python -m clover.main -I {output_file} -L {ref_len} -T {len(valid_tags)}")


if __name__ == '__main__':
    preprocess(
        input_file='output.txt',
        output_file='clover_input.txt',
        ref_len=196,
        margin=5,
        min_reads_per_cluster=5,
    )