#!/usr/bin/env python3
import sys
import os

# === 第一步：确认脚本在运行 ===
print("="*70, file=sys.stderr, flush=True)
print("🚀 脚本启动", file=sys.stderr, flush=True)
print(f"Python: {sys.version}", file=sys.stderr, flush=True)
print(f"当前目录: {os.getcwd()}", file=sys.stderr, flush=True)
print("="*70, file=sys.stderr, flush=True)

# === 第二步：列出当前目录的文件 ===
print("\n📁 当前目录文件:", file=sys.stderr, flush=True)
try:
    files = os.listdir('')
    for f in files:
        print(f"   {f}", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ❌ 错误: {e}", file=sys.stderr, flush=True)

# === 第三步：导入模块 ===
print("\n📚 导入模块...", file=sys.stderr, flush=True)
try:
    import csv
    print("   ✅ csv", file=sys.stderr, flush=True)
    import ast
    print("   ✅ ast", file=sys.stderr, flush=True)
    import re
    print("   ✅ re", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ❌ 导入失败: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# === 第四步：设置文件路径 ===
print("\n⚙️  设置文件路径...", file=sys.stderr, flush=True)

# 🔴 改这里 - 填入你的完整路径
CLOVER_OUTPUT_FILE = '/Users/miemie/Clover/Clover/output/output_20251111_190435.txt.txt'
ORIGINAL_READS_FILE = '/Users/miemie/Clover/Clover/example/example_index_data.txt'
METADATA_OUTPUT_FILE = './metadata.csv'

print(f"   Clover 输出: {CLOVER_OUTPUT_FILE}", file=sys.stderr, flush=True)
print(f"   原始文件: {ORIGINAL_READS_FILE}", file=sys.stderr, flush=True)
print(f"   输出文件: {METADATA_OUTPUT_FILE}", file=sys.stderr, flush=True)

# === 第五步：检查文件是否存在 ===
print("\n🔍 检查文件是否存在...", file=sys.stderr, flush=True)

if os.path.exists(CLOVER_OUTPUT_FILE):
    size = os.path.getsize(CLOVER_OUTPUT_FILE)
    print(f"   ✅ {CLOVER_OUTPUT_FILE} ({size} 字节)", file=sys.stderr, flush=True)
else:
    print(f"   ❌ {CLOVER_OUTPUT_FILE} 不存在！", file=sys.stderr, flush=True)
    print(f"      请检查路径或文件名", file=sys.stderr, flush=True)

if os.path.exists(ORIGINAL_READS_FILE):
    size = os.path.getsize(ORIGINAL_READS_FILE)
    print(f"   ✅ {ORIGINAL_READS_FILE} ({size} 字节)", file=sys.stderr, flush=True)
else:
    print(f"   ❌ {ORIGINAL_READS_FILE} 不存在！", file=sys.stderr, flush=True)
    print(f"      将跳过原始文件解析", file=sys.stderr, flush=True)

# === 第六步：读取 Clover 文件 ===
print("\n📖 读取 Clover 输出文件...", file=sys.stderr, flush=True)

try:
    with open(CLOVER_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    print(f"   ✅ 文件读取成功 ({len(content)} 字符)", file=sys.stderr, flush=True)
    print(f"   文件内容预览（前300字符）:", file=sys.stderr, flush=True)
    print(f"   {content[:300]}", file=sys.stderr, flush=True)
    
    # 尝试解析
    print(f"\n   尝试解析为列表...", file=sys.stderr, flush=True)
    try:
        index_list = ast.literal_eval(content)
        print(f"   ✅ 解析成功！获得 {len(index_list)} 条记录", file=sys.stderr, flush=True)
        
        # 显示前几条
        print(f"\n   前5条记录:", file=sys.stderr, flush=True)
        for i, item in enumerate(index_list[:5]):
            print(f"      {i+1}. {item}", file=sys.stderr, flush=True)
    
    except Exception as e:
        print(f"   ❌ 解析失败: {e}", file=sys.stderr, flush=True)

except Exception as e:
    print(f"   ❌ 读取失败: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)

print("\n" + "="*70, file=sys.stderr, flush=True)
print("诊断完成", file=sys.stderr, flush=True)
print("="*70, file=sys.stderr, flush=True)
