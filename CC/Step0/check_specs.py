import os
import sys
import subprocess
import re
import multiprocessing

def run_cmd(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        return output
    except Exception as e:
        return f"Error: {e}"

def get_cpu_info():
    print(f"\n{'='*20} CPU 信息 {'='*20}")
    # 核心数
    phys_cores = run_cmd("grep 'physical id' /proc/cpuinfo | sort -u | wc -l")
    cpu_cores = multiprocessing.cpu_count()
    model = run_cmd("grep 'model name' /proc/cpuinfo | head -n 1").split(':')[-1].strip()
    
    print(f"CPU 型号: {model}")
    print(f"逻辑核数: {cpu_cores}")
    
    # 负载
    load_avg = os.getloadavg()
    print(f"当前负载: {load_avg} (1min, 5min, 15min)")
    print(f"建议上限: 负载 < {cpu_cores} 时系统流畅")

def get_mem_info():
    print(f"\n{'='*20} 内存 (RAM) 信息 {'='*20}")
    # 使用 free -h
    mem_str = run_cmd("free -g")
    lines = mem_str.split('\n')
    if len(lines) >= 2:
        headers = lines[0].split()
        values = lines[1].split()
        # total, used, free, shared, buff/cache, available
        total = values[1]
        avail = values[-1]
        print(f"总内存:   {total} GB")
        print(f"当前可用: {avail} GB (这是决定你能否并发的关键)")
    else:
        print("无法读取内存信息")

def get_gpu_info():
    print(f"\n{'='*20} GPU 显卡信息 {'='*20}")
    try:
        # 简单信息
        gpu_name = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader")
        gpu_mem = run_cmd("nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader")
        gpu_util = run_cmd("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader")
        
        print(f"显卡型号: {gpu_name}")
        print(f"显存详情: {gpu_mem} (Total, Free, Used)")
        print(f"GPU利用率: {gpu_util}")
        
        # 详细显存进程
        print("\n--- 正在占用显卡的进程 ---")
        os.system("nvidia-smi") 
    except:
        print("未检测到 NVIDIA 驱动或 GPU")

def get_disk_info():
    print(f"\n{'='*20} 磁盘空间 {'='*20}")
    os.system("df -h .")

def check_process_conflict():
    print(f"\n{'='*20} 潜在的大户进程 {'='*20}")
    # 查找 python 和 clover 相关进程
    print("Top 5 内存占用进程:")
    os.system("ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 6")

if __name__ == "__main__":
    print(f"🚀 系统资源侦察报告")
    get_cpu_info()
    get_mem_info()
    get_gpu_info()
    get_disk_info()
    check_process_conflict()
    print(f"\n{'='*50}")