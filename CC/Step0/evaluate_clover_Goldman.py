import os
import sys
import array
from collections import defaultdict, Counter

# ================= 配置区域 =================
# 1. 原始带标签的数据文件 (Ground Truth)
# Goldman 数据集的真实标签文件
GT_FILE = "给师妹的clover数据集/now_goldman_tags_reads.txt"

# 2. Clover 聚类输出文件 (Prediction)
# 对应 run_real_data_Goldman.py 生成的输出位置
CLOVER_OUT_FILE = "/hy-tmp/code/CC/Step0/Experiments/Goldman_Real/02_CloverOut/clover_result_merged.txt"

# ===========================================

class CompactGroundTruth:
    def __init__(self):
        self.tag_to_id = {}    # 唯一 Tag (str) -> int ID
        self.id_to_tag = []    # int ID -> 唯一 Tag (str)
        # 使用 'I' (unsigned int) 数组存储，极省内存
        self.read_labels = array.array('I') 
        
    def load(self, file_path):
        print(f"📖 [1/3] 正在加载 Ground Truth (Goldman)...")
        if not os.path.exists(file_path):
            print(f"❌ 错误: 找不到文件 {file_path}")
            sys.exit(1)

        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    self.read_labels.append(0)
                    continue
                
                # Goldman 的 tag 是第一列 (例如 "4022", "153133")
                tag = parts[0]
                
                if tag not in self.tag_to_id:
                    new_id = len(self.id_to_tag) + 1 
                    self.tag_to_id[tag] = new_id
                    self.id_to_tag.append(tag)
                
                self.read_labels.append(self.tag_to_id[tag])
                
                if (i + 1) % 5000000 == 0:
                    print(f"   已索引 {i + 1} 行...", end='\r')
                        
        print(f"\n   ✅ GT 加载完成。总 Reads: {len(self.read_labels)}")
        print(f"   - 唯一 Tags 数: {len(self.id_to_tag)}")
        
        # 打印内存占用
        mem_mb = self.read_labels.buffer_info()[1] * self.read_labels.itemsize / (1024*1024)
        print(f"   - 标签数组内存占用: {mem_mb:.2f} MB")

    def get_tag_id(self, read_idx):
        if read_idx < 0 or read_idx >= len(self.read_labels):
            return 0
        return self.read_labels[read_idx]


def stream_clover_tokens(file_path, chunk_size=1024*1024):
    """
    流式读取生成器：
    清洗所有标点符号和引号，只产出有效的数据 token
    """
    with open(file_path, 'r') as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # 彻底清洗：去除 [] () , ' "
            cleaned_chunk = chunk.replace('[', ' ').replace(']', ' ')\
                                 .replace('(', ' ').replace(')', ' ')\
                                 .replace(',', ' ')\
                                 .replace("'", " ").replace('"', " ")
            
            buffer += cleaned_chunk
            tokens = buffer.split()
            
            # 判断 buffer 末尾是否完整
            if chunk[-1].isspace() or chunk[-1] in "[](),'\"": 
                for token in tokens:
                    yield token
                buffer = ""
            else:
                if len(tokens) > 0:
                    for token in tokens[:-1]:
                        yield token
                    buffer = tokens[-1]
                else:
                    pass
        
        if buffer.strip():
            yield buffer.strip()

def evaluate(gt_data, clover_path):
    print(f"\n🚀 [2/3] 流式处理聚类结果 (Token Stream)...")
    
    cluster_stats = defaultdict(Counter)
    total_reads_clustered = 0
    
    # 获取 token 流生成器
    token_stream = stream_clover_tokens(clover_path)
    
    try:
        # 每次取两个 token：(index, cluster_id)
        while True:
            try:
                idx_token = next(token_stream)
                cid_token = next(token_stream)
            except StopIteration:
                break
            
            line_idx_1based = int(idx_token)
            cluster_id = cid_token 
            
            # 过滤 Clover 的噪声标记 (-1)
            if str(cluster_id) == '-1':
                continue
                
            # 转换索引 (1-based -> 0-based)
            read_idx = line_idx_1based - 1
            
            # 获取真实 Tag ID
            true_tag_id = gt_data.get_tag_id(read_idx)
            
            if true_tag_id != 0:
                cluster_stats[str(cluster_id)][true_tag_id] += 1
                total_reads_clustered += 1
                
            if total_reads_clustered % 5000000 == 0 and total_reads_clustered > 0:
                print(f"   已统计 {total_reads_clustered} 个聚类成员...", end='\r')

    except ValueError as e:
        print(f"\n⚠️ 解析警告: 数据格式异常: {e}")

    # === [3/3] 生成报告 ===
    print(f"\n\n📊 正在汇总统计信息...")
    
    correct_reads_count = 0
    recovered_tag_ids = set()
    cluster_purities = []
    
    for cid, counts in cluster_stats.items():
        if not counts: continue
        
        # 找出该簇的主导 Tag
        dominant_tag_id, dominant_count = counts.most_common(1)[0]
        total_in_cluster = sum(counts.values())
        
        purity = dominant_count / total_in_cluster
        cluster_purities.append(purity)
        
        correct_reads_count += dominant_count
        recovered_tag_ids.add(dominant_tag_id)

    total_unique_tags = len(gt_data.id_to_tag)
    avg_purity = sum(cluster_purities) / len(cluster_purities) if cluster_purities else 0
    micro_accuracy = correct_reads_count / total_reads_clustered if total_reads_clustered else 0
    recovery_rate = len(recovered_tag_ids) / total_unique_tags if total_unique_tags else 0
    
    print("\n" + "="*40)
    print("       📊 CLOVER 评估报告 - Goldman 数据集")
    print("="*40)
    print(f"1. 原始 Tag 恢复率 (Recovery Rate):")
    print(f"   {len(recovered_tag_ids)} / {total_unique_tags}  ({recovery_rate*100:.2f}%)")
    print(f"   (有多少个原始文件被成功找回)")
    
    print(f"\n2. 平均簇纯度 (Average Purity):")
    print(f"   {avg_purity*100:.2f}%")
    print(f"   (簇内部的一致性)")
    
    print(f"\n3. 整体准确率 (Micro Accuracy):")
    print(f"   {micro_accuracy*100:.2f}%")
    
    print(f"\n4. 统计摘要:")
    print(f"   有效聚类 Reads 数: {total_reads_clustered}")
    print(f"   生成的簇数量: {len(cluster_stats)}")
    print("="*40)

if __name__ == "__main__":
    if not os.path.exists(CLOVER_OUT_FILE):
        print(f"❌ 找不到 Clover 输出文件: {CLOVER_OUT_FILE}")
        print(f"   请先运行 run_real_data_Goldman.py 生成结果。")
        sys.exit(1)

    gt = CompactGroundTruth()
    gt.load(GT_FILE)
    evaluate(gt, CLOVER_OUT_FILE)