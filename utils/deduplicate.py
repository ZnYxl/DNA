# utils/deduplicate.py
import os

def load_fasta(path):
    """读取FASTA文件"""
    seqs = []
    if not os.path.exists(path): return seqs
    
    with open(path, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    seqs.append(parse_entry(header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            seqs.append(parse_entry(header, "".join(seq_lines)))
    return seqs

def parse_entry(header, seq):
    # 解析 Header: >cluster_0_reads50_highconf40...
    parts = header[1:].split('_')
    meta = {'header': header, 'seq': seq, 'count': 1, 'id': -1}
    
    try:
        for p in parts:
            if p.startswith('reads'):
                meta['count'] = int(p.replace('reads', ''))
            elif p.startswith('cluster'):
                meta['id'] = int(p.replace('cluster', ''))
    except:
        pass
    return meta

def run_deduplication(input_fasta, output_fasta):
    print(f"\n🧹 [Post-Process] 开始序列去重 (Deduplication)...")
    print(f"   📂 输入: {input_fasta}")
    
    entries = load_fasta(input_fasta)
    if not entries:
        print("   ❌ 文件不存在或为空")
        return

    print(f"   📊 原始簇数量: {len(entries)}")
    
    # 1. 按 Reads 数量降序排序 (保留大簇的 Header)
    entries.sort(key=lambda x: x['count'], reverse=True)
    
    unique_seqs = {} 
    merged_count = 0
    
    for entry in entries:
        seq = entry['seq']
        # 精确去重策略：序列完全一样才合并
        if seq in unique_seqs:
            merged_count += 1
            continue
        unique_seqs[seq] = entry

    final_entries = list(unique_seqs.values())
    final_entries.sort(key=lambda x: x['id']) # 按ID排序方便查看
    
    print(f"   📝 正在写入...")
    with open(output_fasta, 'w') as f:
        for e in final_entries:
            f.write(f"{e['header']}\n{e['seq']}\n")
            
    print(f"   ✅ 去重完成!")
    print(f"      - 合并冗余: {merged_count}")
    print(f"      - 最终簇数: {len(final_entries)}")
    print(f"   💾 输出保存: {output_fasta}")