"""
run_real_data.py
================
处理真实 DNA 数据集的流水线（ERR036/Fountain, Goldman, ODNA）。
修复版 (v2 - 中文日志)：
1. 内存优化：全程流式处理 FASTA，解决 8GB+ 数据集 OOM 问题。
2. 格式修复：适配 Clover 输出的 Python List 格式 ([('id','id'),...])，使用 mmap 解析。
"""

import os
import sys
import subprocess
import collections
import re
import mmap

# 确保能 import 用户自己的 utils.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import utils
except ImportError:
    print("⚠️  警告: 未找到 utils.py。部分功能可能缺失。")

# ==============================================================================
# 配置区
# ==============================================================================

DATA_DIR = "/hy-tmp/code/CC/Step0/给师妹的clover数据集"
CLOVER_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Clover")

DATASETS = {
    "ERR036":  {"fasta": "ERR036_fa.fasta"},
    "Goldman": {"fasta": "Goldman_fa.fasta"},
    "ODNA":    {"fasta": "ODNA_fa.fasta"},
}

CLOVER_PROCESSES = 0  # 0=单进程
USE_LOW_MEMORY = True # 开启低内存模式

# ==============================================================================
# 核心工具：流式 FASTA 处理
# ==============================================================================

def yield_fasta_records(fasta_path):
    """
    生成器：逐条读取 FASTA，不占用内存。
    yield (header_id, sequence_str)
    """
    with open(fasta_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header:
                    yield header, "".join(seq_parts).upper()
                # 提取 ID：取 > 后面的第一个字符串
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        # 最后一条
        if header:
            yield header, "".join(seq_parts).upper()

def analyze_length_distribution_streaming(fasta_path):
    """第一遍扫描：统计长度分布"""
    counter = collections.Counter()
    print(f"    正在扫描长度分布...")
    for idx, (_, seq) in enumerate(yield_fasta_records(fasta_path)):
        counter[len(seq)] += 1
        if (idx + 1) % 5000000 == 0:
            print(f"      已扫描 {idx + 1:,} 条序列...")
    return counter

def filter_and_write_streaming(fasta_path, output_path, target_length):
    """第二遍扫描：过滤并写入 raw_reads.txt"""
    count = 0
    with open(output_path, 'w') as f_out:
        for idx, (rid, seq) in enumerate(yield_fasta_records(fasta_path)):
            if len(seq) == target_length:
                f_out.write(f"{rid}\t{seq}\n")
                count += 1
            if (idx + 1) % 5000000 == 0:
                print(f"      已处理 {idx + 1:,} 条序列...")
    return count

# ==============================================================================
# 核心工具：Clover 运行与解析
# ==============================================================================

def run_clover(raw_reads_path, clover_out_dir, output_basename, seq_length):
    """运行 Clover"""
    env = os.environ.copy()
    env["PYTHONPATH"] = CLOVER_REPO + os.pathsep + env.get("PYTHONPATH", "")
    os.makedirs(clover_out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "clover.main",
        "-I", os.path.abspath(raw_reads_path),
        "-O", output_basename,
        "-L", str(seq_length),
        "-P", str(CLOVER_PROCESSES),
        "--no-tag"
    ]
    if USE_LOW_MEMORY:
        cmd.append("--low")

    print(f"    执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=clover_out_dir)

def parse_clover_python_list_format(file_path):
    """
    使用 mmap 解析巨大的 Python List 字符串文件。
    格式: [('id1', 'id2'), ('id3', 'id4')...]
    yield (read_id, cluster_id)
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return

    # 正则匹配 ('id1', 'id2') 允许中间有空格
    # Encode pattern to bytes for mmap
    pattern = re.compile(rb"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)")

    with open(file_path, 'r+b') as f:
        # 使用 mmap 避免加载整个文件
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for match in pattern.finditer(mm):
                # 提取 bytes 并解码为 str
                r_id = match.group(1).decode('utf-8')
                c_id = match.group(2).decode('utf-8')
                yield r_id, c_id

def clover_to_feddna_streaming(clover_result_path, raw_reads_path, output_dir):
    """
    流式转换：支持 Clover 的 Python List 输出格式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 扫描 Clover 结果 (cluster -> [reads])
    print(f"    [1/3] 解析 Clover 结果 (使用 mmap)...")
    cluster_members = collections.defaultdict(list)
    count = 0
    
    # 自动判断是 list 格式还是 tab 格式
    # 读取前几个字节判断
    is_list_format = False
    with open(clover_result_path, 'r') as f:
        start = f.read(10)
        if start.strip().startswith('['):
            is_list_format = True

    if is_list_format:
        print("      检测到 Python 列表格式 (例如 [('id', 'id')...])")
        iterator = parse_clover_python_list_format(clover_result_path)
    else:
        print("      检测到 TSV 格式")
        # 简单的生成器适配器
        def tsv_iter():
            with open(clover_result_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2: yield parts[0], parts[1]
        iterator = tsv_iter()

    for r_id, c_id in iterator:
        cluster_members[c_id].append(r_id)
        count += 1
        if count % 1000000 == 0:
            print(f"      已发现 {count:,} 对匹配...")

    num_clusters = len(cluster_members)
    print(f"    ✅ 发现 {num_clusters:,} 个簇，共分配 {count:,} 条 reads。")
    
    if num_clusters == 0:
        print("    ⚠️ 警告: 未发现任何簇！请检查输入文件格式。")
        return 0, output_dir

    # 2. 索引需要的 Read 序列
    print(f"    [2/3] 从 raw_reads 索引序列...")
    needed_reads = set()
    for members in cluster_members.values():
        needed_reads.update(members)
    
    read_seqs = {}
    with open(raw_reads_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                rid, seq = parts
                if rid in needed_reads:
                    read_seqs[rid] = seq
    
    print(f"    ✅ 已索引 {len(read_seqs):,} 条序列。")

    # 3. 写入 FedDNA
    print(f"    [3/3] 写入 FedDNA 输入文件...")
    read_file = os.path.join(output_dir, "read.txt")
    ref_file = os.path.join(output_dir, "ref.txt")
    
    with open(read_file, 'w') as f_read, open(ref_file, 'w') as f_ref:
        for c_id, members in cluster_members.items():
            # 获取序列
            seqs = [read_seqs[rid] for rid in members if rid in read_seqs]
            if not seqs: continue
            
            # 写入 Reference (Center)
            # 注意：Clover 的 c_id 有时就是 center read id，有时是虚拟的
            # 这里简单取簇中第一条作为 ref，或者如果你确信 c_id 是 read id 且在 seqs 里，可以用它
            center_seq = seqs[0]
            if c_id in read_seqs:
                center_seq = read_seqs[c_id]
            
            f_ref.write(center_seq + "\n")
            for s in seqs:
                f_read.write(s + "\n")
            f_read.write("===============================\n")

    return num_clusters, output_dir

# ==============================================================================
# 主逻辑
# ==============================================================================

def process_dataset(dataset_name, fasta_filename, output_base_dir):
    print(f"\n{'='*60}\n  数据集: {dataset_name}\n{'='*60}")
    
    fasta_path = os.path.join(DATA_DIR, fasta_filename)
    if not os.path.exists(fasta_path):
        print(f"  ❌ 文件未找到: {fasta_path}")
        return None

    base_dir = os.path.join(output_base_dir, dataset_name)
    dir_raw = os.path.join(base_dir, "01_RawData")
    dir_clover = os.path.join(base_dir, "02_CloverOut")
    dir_feddna = os.path.join(base_dir, "03_FedDNA_In")
    for d in [dir_raw, dir_clover, dir_feddna]:
        os.makedirs(d, exist_ok=True)

    # Step 1 & 2: 流式统计长度与过滤
    # ---------------------------
    raw_reads_path = os.path.join(dir_raw, "raw_reads.txt")
    
    # 只有当 raw_reads.txt 不存在或想要重新跑时才执行
    if not os.path.exists(raw_reads_path) or os.path.getsize(raw_reads_path) == 0:
        print("  [步骤 1-2] 分析与过滤 (流式)...")
        length_dist = analyze_length_distribution_streaming(fasta_path)
        if not length_dist:
            print("  ❌ 未发现 reads。")
            return None
            
        modal_length, count = length_dist.most_common(1)[0]
        print(f"    众数长度 (Modal Length): {modal_length} bp (数量: {count})")
        
        kept_count = filter_and_write_streaming(fasta_path, raw_reads_path, modal_length)
        print(f"  ✅ 已写入 {kept_count} 条 reads 到 {raw_reads_path}")
    else:
        print(f"  提示: raw_reads.txt 已存在，跳过过滤步骤。")
        # 简单读取第一行获取长度（假设一致）
        with open(raw_reads_path) as f:
            line = f.readline()
            if line:
                modal_length = len(line.split('\t')[1].strip())
            else:
                modal_length = 0 # 异常处理
        kept_count = "未知 (已缓存)"

    # Step 3: 运行 Clover
    # ---------------------------
    print(f"  [步骤 3] 运行 Clover (L={modal_length})...")
    output_basename = "clover_result"
    
    # 检查结果文件位置
    if USE_LOW_MEMORY:
        clover_result_file = os.path.join(dir_clover, "all_" + output_basename + ".txt")
    else:
        clover_result_file = os.path.join(dir_clover, output_basename + ".txt")
    
    if not os.path.exists(clover_result_file) or os.path.getsize(clover_result_file) == 0:
        try:
            run_clover(raw_reads_path, dir_clover, output_basename, modal_length)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Clover 运行失败: {e}")
            return None
    else:
        print(f"  ✅ Clover 输出已存在: {clover_result_file}")

    # Step 4: 转换 FedDNA (修复格式解析)
    # ---------------------------
    print(f"  [步骤 4] 转换为 FedDNA 格式...")
    clusters, _ = clover_to_feddna_streaming(clover_result_file, raw_reads_path, dir_feddna)
    
    return {"dataset": dataset_name, "clusters": clusters}

def run():
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Experiments")
    print("🚀 流水线开始执行")
    
    for name, cfg in DATASETS.items():
        process_dataset(name, cfg["fasta"], output_base)

if __name__ == "__main__":
    run()