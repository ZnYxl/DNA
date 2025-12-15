import csv
import ast
import re
import os
from collections import defaultdict
from pathlib import Path

# ============================================================================
# 核心函数 1: 从原始DNA文件提取所有 Read ID
# ============================================================================
def get_all_input_read_ids(filepath):
    """
    从 FastA/TXT/FastQ 文件提取所有 Read ID
    支持格式：
    - FastA: >seq_id
    - FastQ: @seq_id
    - TXT: seq_id SEQUENCE
    """
    all_ids = set()
    
    if not os.path.exists(filepath):
        print(f"警告: 输入文件不存在 {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # FastA 格式: >seq_id
                if line.startswith('>'):
                    read_id = line[1:].split()[0].strip()
                    if read_id:
                        all_ids.add(read_id)
                
                # FastQ 格式: @seq_id
                elif line.startswith('@'):
                    read_id = line[1:].split()[0].strip()
                    if read_id:
                        all_ids.add(read_id)
                
                # TXT 格式: seq_id SEQUENCE (或其他带空格的格式)
                else:
                    parts = line.split()
                    if len(parts) >= 1:
                        read_id = parts[0].strip()
                        if read_id and not line.startswith('#'):  # 跳过注释行
                            all_ids.add(read_id)
        
        print(f"✅ 成功读取输入文件: {filepath}")
        print(f"   发现 {len(all_ids)} 个 Read ID")
        return all_ids
    
    except Exception as e:
        print(f"❌ 错误: 无法读取输入文件。{e}")
        return None


# ============================================================================
# 核心函数 2: 从 Clover 输出文件提取聚类结果
# ============================================================================
def parse_clover_output(clover_output_path):
    """
    解析 Clover 输出文件
    支持格式：
    - 纯列表: [('seq0', 'seq1'), ('seq2', 'seq3'), ...]
    - 带日志: 前面有文本，然后是列表
    """
    
    if not os.path.exists(clover_output_path):
        print(f"❌ 错误: Clover 输出文件不存在 {clover_output_path}")
        return None
    
    try:
        with open(clover_output_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        print(f"✅ 成功读取 Clover 输出文件: {clover_output_path}")
        print(f"   文件大小: {len(content)} 字符")
        
        # 方法 1: 直接尝试 eval（如果是纯列表）
        try:
            index_list = ast.literal_eval(content)
            print(f"   ✓ 直接解析成功，获得 {len(index_list)} 条聚类关系")
            return index_list
        except (ValueError, SyntaxError):
            pass
        
        # 方法 2: 使用正则表达式提取列表部分
        # 匹配 [(...)，(...)，...] 的模式
        match = re.search(r'\[\s*\([^)]*\).*?\]', content, re.DOTALL)
        if match:
            list_str = match.group(0)
            index_list = ast.literal_eval(list_str)
            print(f"   ✓ 正则表达式提取成功，获得 {len(index_list)} 条聚类关系")
            return index_list
        
        # 方法 3: 寻找包含元组的行
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    index_list = ast.literal_eval(line)
                    print(f"   ✓ 行扫描提取成功，获得 {len(index_list)} 条聚类关系")
                    return index_list
                except (ValueError, SyntaxError):
                    continue
        
        print(f"❌ 错误: 无法解析 Clover 输出文件。请检查文件格式。")
        print(f"   文件前 500 字符内容:\n{content[:500]}")
        return None
    
    except Exception as e:
        print(f"❌ 错误: 读取 Clover 文件失败。{e}")
        return None


# ============================================================================
# 核心函数 3: 生成 metadata.csv
# ============================================================================
def generate_metadata_csv(clover_output_path, input_reads_file, output_filename="metadata.csv"):
    """
    生成最终的 metadata.csv 文件
    """
    
    print("\n" + "="*70)
    print("开始生成 metadata.csv")
    print("="*70 + "\n")
    
    # 步骤 1: 解析 Clover 输出
    print("[步骤 1] 解析 Clover 输出文件...")
    index_list = parse_clover_output(clover_output_path)
    if index_list is None:
        print("❌ 失败：无法解析 Clover 输出文件")
        return {}
    
    # 步骤 2: 获取所有输入 Read ID
    print("\n[步骤 2] 读取原始输入文件...")
    all_input_reads = get_all_input_read_ids(input_reads_file)
    
    # 步骤 3: 构建聚类映射表
    print("\n[步骤 3] 构建聚类映射表...")
    cluster_map = {}  # follower -> founder
    all_founders = set()
    
    # 验证 index_list 格式
    if not index_list:
        print("⚠️  警告: Clover 输出为空")
        index_list = []
    
    for entry in index_list:
        try:
            if isinstance(entry, tuple) and len(entry) == 2:
                follower_id, founder_id = entry
                # 提取纯 ID（去掉可能的 _clusterX 后缀）
                if isinstance(follower_id, str) and isinstance(founder_id, str):
                    cluster_map[follower_id] = founder_id
                    all_founders.add(founder_id)
        except Exception as e:
            print(f"   ⚠️  警告: 无法处理条目 {entry}。错误: {e}")
            continue
    
    print(f"   ✓ 读取 {len(cluster_map)} 条 follower-founder 映射")
    print(f"   ✓ 发现 {len(all_founders)} 个 Founder")
    
    # 步骤 4: 补全映射表（包括 Founder 自映射和孤儿 ID）
    print("\n[步骤 4] 补全映射表...")
    final_mapping = {}
    orphan_count = 0
    
    if all_input_reads:
        # 情况 1: 能成功读取输入文件
        for read_id in all_input_reads:
            if read_id in cluster_map:
                # 这是一个 follower
                final_mapping[read_id] = cluster_map[read_id]
            elif read_id in all_founders:
                # 这是一个 founder（自映射）
                final_mapping[read_id] = read_id
            else:
                # 这是一个孤儿（未被匹配）
                final_mapping[read_id] = '-1'
                orphan_count += 1
        
        print(f"   ✓ 处理了 {len(all_input_reads)} 个输入 Read ID")
        print(f"   ✓ 其中孤儿 ID: {orphan_count} 个")
    else:
        # 情况 2: 无法读取输入文件，仅使用 Clover 输出
        print("   ⚠️  未能读取输入文件，仅使用 Clover 输出的 ID")
        
        # 添加所有 founder（自映射）
        for founder_id in all_founders:
            final_mapping[founder_id] = founder_id
        
        # 添加所有 follower 的映射
        for follower_id, founder_id in cluster_map.items():
            final_mapping[follower_id] = founder_id
    
    # 步骤 5: 写入 CSV
    print("\n[步骤 5] 写入 CSV 文件...")
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['read_id', 'cluster_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 按 read_id 排序后写入
            sorted_reads = sorted(list(final_mapping.keys()))
            for read_id in sorted_reads:
                writer.writerow({
                    'read_id': read_id,
                    'cluster_id': final_mapping[read_id]
                })
        
        print(f"   ✓ CSV 文件已写入: {output_filename}")
    except Exception as e:
        print(f"   ❌ 错误: 无法写入 CSV 文件。{e}")
        return {}
    
    # 步骤 6: 打印统计信息
    print("\n" + "="*70)
    print("统计信息")
    print("="*70)
    print(f"总 Read ID 数: {len(final_mapping)}")
    print(f"Founder 总数: {len(all_founders)}")
    print(f"Follower 总数: {len(cluster_map)}")
    print(f"孤儿 ID 数: {orphan_count}")
    print(f"聚类覆盖率: {(len(cluster_map) + len(all_founders)) / len(final_mapping) * 100:.2f}%")
    print("="*70 + "\n")
    
    print(f"✅ metadata.csv 生成成功！\n")
    
    return final_mapping


# ============================================================================
# 主函数
# ============================================================================
if __name__ == '__main__':
    
    print("DNA 聚类 Metadata 生成脚本 ")

    
    # ======= 实际文件路径 =======
    CLOVER_OUTPUT_FILE = 'output_20251111_190435.txt.txt'  # Clover 输出文件
    ORIGINAL_READS_FILE = 'example_index_data.txt'  # 原始 DNA 文件
    METADATA_OUTPUT_FILE = 'metadata.csv'  # 输出 CSV 文件
    
    # ================================================
    
    print("📋 配置信息:")
    print(f"   Clover 输出文件: {CLOVER_OUTPUT_FILE}")
    print(f"   原始 Reads 文件: {ORIGINAL_READS_FILE}")
    print(f"   输出 CSV 文件: {METADATA_OUTPUT_FILE}\n")
    
    # 检查文件是否存在
    if not os.path.exists(CLOVER_OUTPUT_FILE):
        print(f"❌ 错误: 找不到 Clover 输出文件: {CLOVER_OUTPUT_FILE}")
        print(f"   请确保文件存在，或修改脚本中的 CLOVER_OUTPUT_FILE 路径\n")
        exit(1)
    
    if not os.path.exists(ORIGINAL_READS_FILE):
        print(f"⚠️  警告: 找不到原始 Reads 文件: {ORIGINAL_READS_FILE}")
        print(f"   将仅使用 Clover 输出进行处理\n")
    
    # 生成 metadata.csv
    result = generate_metadata_csv(
        clover_output_path=CLOVER_OUTPUT_FILE,
        input_reads_file=ORIGINAL_READS_FILE,
        output_filename=METADATA_OUTPUT_FILE
    )
    
    if result:
        print("✨ 所有步骤已完成！")
        print(f"📁 请查看输出文件: {METADATA_OUTPUT_FILE}\n")
    else:
        print("❌ 处理过程中出现错误，请检查日志信息\n")
