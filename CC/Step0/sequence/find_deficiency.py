# # 读取文件内容
# with open('output.txt', 'r') as file:
#     lines = file.readlines()
#
# # 提取每行前面的数字
# numbers_in_file = set()
# for line in lines:
#     number = int(line.split('\t')[0])
#     numbers_in_file.add(number)
#
# # 生成1到11826的完整数字范围
# full_range = set(range(1, 210000))
#
# # 找出缺失的数字
# missing_numbers = sorted(full_range - numbers_in_file)
#
# # 输出缺失的数字
# print(f"Missing numbers from 1 to 210000: {missing_numbers}")
# print(f"Total missing numbers: {len(missing_numbers)}")


import pandas as pd

# 读取Excel文件
excel_file = r'/amax/linhaiyan_data/P10_5_BDDP210000009/41467_2022_33046_MOESM5_ESM.xlsx'
df = pd.read_excel(excel_file, usecols=[0])  # 只读取第一列
excel_numbers = set(df.iloc[:, 0].dropna().astype(int))

# 读取output.txt文件
txt_file = 'output.txt'
with open(txt_file, 'r') as file:
    lines = file.readlines()

txt_numbers = set()
for line in lines:
    number = int(line.split('\t')[0])
    txt_numbers.add(number)

# 找出Excel中但不在txt文件中的数字
missing_numbers = excel_numbers - txt_numbers

# 输出未出现在txt文件中的数字个数
print(f"Missing numbers : {missing_numbers}")
print(f"Count of numbers in Excel but not in txt file: {len(missing_numbers)}")
