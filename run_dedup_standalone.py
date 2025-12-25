import os
import argparse
from collections import defaultdict

def parse_header_meta(header):
    """
    解析 Header 元数据
    格式示例: >cluster_0_reads50_highconf40_strength12.5
    """
    meta = {'header': header, 'count': 0, 'id': -1}
    try:
        parts = header[1:].split('_')
        for p in parts:
            if p.startswith('reads'):
                meta['count'] = int(p.replace('reads', ''))
            elif p.startswith('cluster'):
                meta['id'] = int(p.replace('cluster', ''))
    except:
        pass
    return meta

def load_fasta(path):
    """读取FASTA并解析"""
    entries = []
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return entries
    
    with open(path, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    full_seq = "".join(seq_lines)
                    meta = parse_header_meta(header)
                    meta['seq'] = full_seq
                    entries.append(meta)
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        
        # 处理最后一条
        if header:
            full_seq = "".join(seq_lines)
            meta = parse_header_meta(header)
            meta['seq'] = full_seq
            entries.append(meta)
            
    return entries

def deduplicate(input_path, output_path):
    print(f"🚀 开始执行后处理去重 (Post-processing Deduplication)...")
    print(f"📂 输入文件: {input_path}")
    
    # 1. 加载数据
    entries = load_fasta(input_path)
    if not entries: return
    
    total_before = len(entries)
    print(f"   📊 原始簇数量: {total_before}")
    
    # 2. 排序：按 reads 数量从大到小排序
    # 逻辑：如果有两个相同的序列，我们保留 reads 数多的那个作为主簇
    entries.sort(key=lambda x: x['count'], reverse=True)
    
    # 3. 去重逻辑
    unique_map = {} # seq -> entry
    merged_count = 0
    substring_merged_count = 0
    
    # 先做一遍完全精确匹配
    for entry in entries:
        seq = entry['seq']
        if seq not in unique_map:
            unique_map[seq] = entry
        else:
            # 发现完全一样的序列，视为冗余，直接丢弃（或合并计数）
            merged_count += 1
            # 可选：如果你想累加 reads 数，可以在这里做
            # unique_map[seq]['count'] += entry['count']

    # (可选) 做一遍子串合并：如果长序列包含了短序列，且短序列很可能是碎片
    # 为了安全，这里只处理非常明显的包含关系
    # 注意：这步复杂度较高 O(N^2)，如果只有1万条数据很快，但如果有几十万条会慢
    # 鉴于我们要保Recall，先只做精确去重，这通常能解决95%的问题
    
    final_entries = list(unique_map.values())
    
    # 恢复按ID排序（可选，为了好看）
    final_entries.sort(key=lambda x: x['id'])
    
    # 4. 保存
    with open(output_path, 'w') as f:
        for e in final_entries:
            f.write(f"{e['header']}\n{e['seq']}\n")
            
    total_after = len(final_entries)
    
    print(f"\n✅ 去重完成!")
    print(f"   📉 消除冗余: {merged_count} ({(merged_count/total_before)*100:.2f}%)")
    print(f"   🏁 最终簇数: {total_after}")
    print(f"   💾 输出保存: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认使用你提供的路径
    default_path = "/mnt/st_data/liangxinyi/code/iterative_results/20251224_155232_Cluster_GT_Test copy/round_3/step2/consensus_sequences.fasta"
    
    parser.add_argument('--input', type=str, default=default_path, help='输入FASTA路径')
    parser.add_argument('--output', type=str, default=None, help='输出FASTA路径 (默认在同目录下加_deduplicated)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.replace(".fasta", "_deduplicated.fasta")
        
    deduplicate(args.input, args.output)