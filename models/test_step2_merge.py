#!/usr/bin/env python3
"""
单独跑 Step 2 + MNN 合并，不走 main_loop，~10 分钟出结果。
用法: python test_step2_merge.py 2>&1 | tee test_merge.log
"""
import argparse
import sys
import os

sys.path.insert(0, '/mnt/st_data/liangxinyi/code')

from models.step2_runner import run_step2

args = argparse.Namespace(
    experiment_dir='/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_NoPrimer/',
    step1_checkpoint='/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_NoPrimer/results/iter_1_step1/models/step1_final_model.pth',
    output_dir='/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_NoPrimer/results/test_merge/',
    dim=256,
    max_length=105,
    batch_size=1024,
    device='cuda',
    round_idx=1,
    refined_labels=None,
    prev_state=None,
    gt_tags_file='/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1_NoPrimer/exp1_tags_reads.txt',
    gt_refs_file=None,
    training_cap=9999000000,
)

print("=" * 60)
print("🧪 单独测试 Step 2 + MNN 合并")
print(f"   Checkpoint: {args.step1_checkpoint}")
print(f"   输出目录:   {args.output_dir}")
print("=" * 60)

results = run_step2(args)

if results:
    print("\n✅ 测试完成！")
    for k, v in results.get('next_round_files', {}).items():
        print(f"   {k}: {v}")
else:
    print("\n❌ Step 2 失败")