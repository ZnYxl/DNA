# utils.py - 改进版
import random
import os
import ast
import collections
import numpy as np

# ==================== 模块1: 改进的数据生成 ====================

def generate_diverse_references(num_clusters, seq_len, min_distance=0.25):
    """
    🔥 生成具有足够区分度的参考序列
    降低min_distance到0.25，适合Clover聚类
    """
    bases = ['A', 'C', 'G', 'T']
    references = []
    max_attempts = 1000
    
    # 生成第一个参考序列
    first_ref = "".join(random.choice(bases) for _ in range(seq_len))
    references.append(first_ref)
    
    # 生成其他参考序列，确保足够的区分度
    for i in range(1, num_clusters):
        attempts = 0
        while attempts < max_attempts:
            candidate = "".join(random.choice(bases) for _ in range(seq_len))
            
            # 检查与已有序列的距离
            min_dist = float('inf')
            for existing_ref in references:
                hamming_dist = sum(c1 != c2 for c1, c2 in zip(candidate, existing_ref))
                hamming_ratio = hamming_dist / seq_len
                min_dist = min(min_dist, hamming_ratio)
            
            if min_dist >= min_distance:
                references.append(candidate)
                break
            attempts += 1
        
        if attempts == max_attempts:
            print(f"⚠️  警告: 簇 {i} 的参考序列可能与其他簇相似度过高")
            # 强制生成一个差异较大的序列
            candidate = list(references[0])
            # 随机改变至少min_distance比例的位置
            positions_to_change = random.sample(range(seq_len), 
                                               int(seq_len * min_distance) + 1)
            for pos in positions_to_change:
                original_base = candidate[pos]
                new_bases = [b for b in bases if b != original_base]
                candidate[pos] = random.choice(new_bases)
            references.append("".join(candidate))
    
    return references

def create_motif_based_references(num_clusters, seq_len):
    """
    🔥 基于motif的参考序列生成 - 更真实的生物学模式
    """
    # 定义一些生物学上有意义的motif
    motifs = [
        "ATCGATCG",      # 简单重复
        "GCGCGCGC",      # GC富集
        "ATATATATAT",    # AT富集
        "CGTACGTA",      # 回文序列
        "TGCATGCA",      # 另一个回文
        "AAGCTTAAGCTT",  # 限制酶位点
        "GAATTCGAATTC",  # EcoRI位点
        "GGATCCGGATCC",  # BamHI位点
        "CCGGCCGG",      # 高GC含量
        "TTAATTAA",      # 低GC含量
    ]
    
    references = []
    for i in range(num_clusters):
        # 选择主要motif
        main_motif = motifs[i % len(motifs)]
        
        # 构建序列
        sequence = []
        pos = 0
        while pos < seq_len:
            if pos + len(main_motif) <= seq_len and random.random() < 0.5:
                # 50%概率插入motif
                sequence.extend(list(main_motif))
                pos += len(main_motif)
            else:
                # 插入随机碱基
                sequence.append(random.choice(['A', 'C', 'G', 'T']))
                pos += 1
        
        # 截断到指定长度
        ref_seq = "".join(sequence[:seq_len])
        
        # 如果太短，用随机序列补齐
        while len(ref_seq) < seq_len:
            ref_seq += random.choice(['A', 'C', 'G', 'T'])
        
        references.append(ref_seq)
    
    return references

def mutate_sequence_realistic(sequence, sub_rate=0.008, del_rate=0.003, ins_rate=0.003):
    """
    🔥 更真实的序列突变模型 - 降低错误率，适合Clover
    """
    bases = ['A', 'C', 'G', 'T']
    result = list(sequence)
    
    i = 0
    while i < len(result):
        # 替换突变 - 模拟真实的转换偏好
        if random.random() < sub_rate:
            original = result[i]
            # 转换偏好：A<->G (嘌呤), C<->T (嘧啶)
            if original == 'A':
                result[i] = 'G' if random.random() < 0.6 else random.choice(['C', 'T'])
            elif original == 'G':
                result[i] = 'A' if random.random() < 0.6 else random.choice(['C', 'T'])
            elif original == 'C':
                result[i] = 'T' if random.random() < 0.6 else random.choice(['A', 'G'])
            elif original == 'T':
                result[i] = 'C' if random.random() < 0.6 else random.choice(['A', 'G'])
        
        # 删除突变
        if random.random() < del_rate:
            result.pop(i)
            continue
        
        # 插入突变
        if random.random() < ins_rate:
            insert_base = random.choice(bases)
            result.insert(i, insert_base)
            i += 1
        
        i += 1
    
    return "".join(result)

def add_quality_variation(reads_data, high_quality_ratio=0.75):
    """
    🔥 添加质量变化 - 适合Clover聚类的质量分布
    """
    enhanced_reads = []
    
    for read_id, sequence, cluster_id, ref_seq in reads_data:
        if random.random() < high_quality_ratio:
            # 高质量read - 很低错误率
            noisy_seq = mutate_sequence_realistic(sequence, 
                                                sub_rate=0.003, 
                                                del_rate=0.001, 
                                                ins_rate=0.001)
            quality = "high"
        else:
            # 低质量read - 中等错误率
            noisy_seq = mutate_sequence_realistic(sequence, 
                                                sub_rate=0.015, 
                                                del_rate=0.008, 
                                                ins_rate=0.008)
            quality = "low"
        
        enhanced_reads.append((read_id, noisy_seq, cluster_id, ref_seq, quality))
    
    return enhanced_reads

def generate_data(output_dir, num_clusters, reads_per_cluster, seq_len, 
                 reference_type="diverse", min_distance=0.25):
    """
    🔥 改进的数据生成函数 - 适合Clover + 神经网络的流水线
    """
    os.makedirs(output_dir, exist_ok=True)
    
    raw_path = os.path.join(output_dir, "raw_reads.txt")
    gt_path = os.path.join(output_dir, "ground_truth.txt")
    stats_path = os.path.join(output_dir, "data_stats.txt")
    
    print(f"🔧 生成改进数据: {num_clusters}簇 x {reads_per_cluster}reads, 长度={seq_len}")
    
    # 1️⃣ 生成高区分度的参考序列
    if reference_type == "motif":
        ground_truths = create_motif_based_references(num_clusters, seq_len)
        print("   使用基于motif的参考序列")
    else:
        ground_truths = generate_diverse_references(num_clusters, seq_len, min_distance)
        print(f"   使用高区分度的随机参考序列 (最小距离={min_distance})")
    
    # 2️⃣ 计算参考序列间的距离统计
    distances = []
    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(ground_truths[i], ground_truths[j]))
            distances.append(hamming_dist / seq_len)
    
    min_distance_actual = min(distances) if distances else 0
    avg_distance = np.mean(distances) if distances else 0
    print(f"   实际簇间距离: 最小={min_distance_actual:.3f}, 平均={avg_distance:.3f}")
    
    # 3️⃣ 生成reads数据
    all_reads_data = []
    counter = 0
    
    for cluster_idx, ref_seq in enumerate(ground_truths):
        for read_idx in range(reads_per_cluster):
            counter += 1
            read_id = f"read_{counter:06d}"
            all_reads_data.append((read_id, ref_seq, cluster_idx, ref_seq))
    
    # 4️⃣ 添加质量变化和突变
    enhanced_reads = add_quality_variation(all_reads_data, high_quality_ratio=0.75)
    
    # 5️⃣ 随机打乱
    random.shuffle(enhanced_reads)
    
    # 6️⃣ 保存数据 - 保持原格式兼容性
    with open(raw_path, 'w') as f:
        for read_id, noisy_seq, cluster_id, ref_seq, quality in enhanced_reads:
            f.write(f"{read_id}\t{noisy_seq}\n")
    
    with open(gt_path, 'w') as f:
        f.write("Read_ID\tCluster_ID\tRef_Seq\tQuality\n")
        for read_id, noisy_seq, cluster_id, ref_seq, quality in enhanced_reads:
            f.write(f"{read_id}\t{cluster_id}\t{ref_seq}\t{quality}\n")
    
    # 7️⃣ 保存统计信息
    with open(stats_path, 'w') as f:
        f.write("=== 改进数据集统计信息 ===\n")
        f.write(f"簇数量: {num_clusters}\n")
        f.write(f"每簇reads数: {reads_per_cluster}\n")
        f.write(f"序列长度: {seq_len}\n")
        f.write(f"总reads数: {len(enhanced_reads)}\n")
        f.write(f"参考序列类型: {reference_type}\n")
        f.write(f"目标最小距离: {min_distance}\n")
        f.write(f"实际最小距离: {min_distance_actual:.3f}\n")
        f.write(f"实际平均距离: {avg_distance:.3f}\n")
        f.write("\n=== 参考序列 ===\n")
        for i, ref in enumerate(ground_truths):
            f.write(f"Cluster_{i}: {ref}\n")
        
        # 质量分布统计
        high_quality_count = sum(1 for _, _, _, _, q in enhanced_reads if q == "high")
        f.write(f"\n=== 质量分布 ===\n")
        f.write(f"高质量reads: {high_quality_count} ({high_quality_count/len(enhanced_reads)*100:.1f}%)\n")
        f.write(f"低质量reads: {len(enhanced_reads)-high_quality_count} ({(len(enhanced_reads)-high_quality_count)/len(enhanced_reads)*100:.1f}%)\n")
        
        # 错误率统计
        f.write(f"\n=== 错误率设置 ===\n")
        f.write(f"高质量reads: 替换0.3%, 插入/删除0.1%\n")
        f.write(f"低质量reads: 替换1.5%, 插入/删除0.8%\n")
    
    print(f"✅ 改进数据已保存:")
    print(f"   Raw reads: {raw_path}")
    print(f"   Ground truth: {gt_path}")
    print(f"   Statistics: {stats_path}")
    
    return raw_path, gt_path

# ==================== 模块2: 格式转换 (保持不变) ====================
def load_raw_reads(file_path):
    d = {}
    with open(file_path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: 
                d[p[0]] = p[1]
    return d

def clover_to_feddna(clover_out_path, raw_reads_path, output_dir):
    """将Clover结果转换为FedDNA格式 - 增强版"""
    raw_reads = load_raw_reads(raw_reads_path)
    clusters = collections.defaultdict(list)
    
    print(f"📊 解析Clover输出: {clover_out_path}")
    
    # 智能解析 Clover 输出
    with open(clover_out_path, 'r') as f:
        content = f.read().strip()
        if content.startswith("[") and content.endswith("]"):
            # 列表格式解析
            try:
                pairs = ast.literal_eval(content)
                for item in pairs:
                    if str(item[1]) not in ['-1', -1]:
                        clusters[str(item[1])].append(str(item[0]))
                print(f"   ✅ 列表格式解析成功，找到 {len(pairs)} 个分配")
            except Exception as e:
                print(f"   ❌ 列表格式解析失败: {e}")
        else:
            # 逐行格式解析
            f.seek(0)
            line_count = 0
            for line in f:
                line_count += 1
                p = line.replace(',', ' ').split()
                if len(p) >= 2 and p[1] not in ['-1', '-1']:
                    clusters[p[1]].append(p[0])
            print(f"   ✅ 逐行格式解析完成，处理 {line_count} 行")
    
    # 统计聚类结果
    valid_clusters = {k: v for k, v in clusters.items() if len(v) > 0}
    cluster_sizes = [len(v) for v in valid_clusters.values()]
    
    print(f"📈 Clover聚类统计:")
    print(f"   有效簇数: {len(valid_clusters)}")
    if cluster_sizes:
        print(f"   簇大小: 最小={min(cluster_sizes)}, 最大={max(cluster_sizes)}, 平均={np.mean(cluster_sizes):.1f}")
    
    # 写入结果
    out_read = os.path.join(output_dir, "read.txt")
    out_ref = os.path.join(output_dir, "reference.txt")
    
    valid_count = 0
    total_reads_written = 0
    
    with open(out_read, 'w') as fr, open(out_ref, 'w') as ff:
        for cid, mems in valid_clusters.items():
            # 选择第一个成员作为伪参考序列
            if mems and mems[0] in raw_reads:
                ff.write(raw_reads[mems[0]] + "\n")  # 伪Reference
                
                # 写入该簇的所有reads
                cluster_read_count = 0
                for m in mems:
                    if m in raw_reads:
                        fr.write(raw_reads[m] + "\n")
                        cluster_read_count += 1
                        total_reads_written += 1
                
                fr.write("===============================\n")
                valid_count += 1
                
                if cluster_read_count > 0:
                    print(f"   簇 {cid}: {cluster_read_count} reads")
    
    print(f"✅ 格式转换完成:")
    print(f"   有效簇数: {valid_count}")
    print(f"   总reads数: {total_reads_written}")
    print(f"   输出文件: {out_read}")
    
    return valid_count, out_read

# ==================== 模块3: 数据验证 ====================
def validate_generated_data(gt_path, raw_path):
    """验证生成数据的质量"""
    try:
        # 读取ground truth
        cluster_stats = collections.defaultdict(int)
        total_reads = 0
        
        with open(gt_path, 'r') as f:
            next(f)  # 跳过header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cluster_id = parts[1]
                    cluster_stats[cluster_id] += 1
                    total_reads += 1
        
        # 读取raw reads
        raw_count = 0
        with open(raw_path, 'r') as f:
            for line in f:
                if line.strip():
                    raw_count += 1
        
        print(f"📊 数据验证结果:")
        print(f"   Ground truth reads: {total_reads}")
        print(f"   Raw reads: {raw_count}")
        print(f"   簇数量: {len(cluster_stats)}")
        print(f"   每簇reads数分布: {dict(collections.Counter(cluster_stats.values()))}")
        
        return total_reads == raw_count and len(cluster_stats) > 0
        
    except Exception as e:
        print(f"❌ 数据验证失败: {e}")
        return False
