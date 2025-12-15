#!/bin/bash
# Clover DNA clustering 优化运行脚本，解决 Too many open files 问题

# 1. 提高文件描述符限制
ulimit -n 4096
if [ $? -ne 0 ]; then
    echo "警告：无法调整打开文件限制，可能导致多进程错误"
fi

# 2. 激活虚拟环境
if [ -d "venv_clover" ]; then
    source venv_clover/bin/activate
    echo "已激活虚拟环境 venv_clover"
else
    echo "未找到虚拟环境，使用系统 Python"
fi

# 3. 安装依赖
pip install -r requirements.txt --quiet

# 4. 设置参数（根据实际机器调低进程数，保证不会打开太多文件）
#INPUT_FILE="example/example_index_data.txt"
INPUT_FILE="/Users/miemie/Library/Mobile Documents/com~apple~CloudDocs/DNA/miemie_DNA/code/raw_reads.txt"
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/output_$(date +%Y%m%d_%H%M%S)"
READ_LENGTH=152
PROCESS_NUM=2
NO_TAG="--no-tag"

echo "Clover开始运行"
echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "参数: -L $READ_LENGTH -P $PROCESS_NUM $NO_TAG"

start_time=$(date +%s)

python -m clover.main -I "$INPUT_FILE" -O "$OUTPUT_FILE" -L "$READ_LENGTH" -P "$PROCESS_NUM" $NO_TAG

exit_code=$?

end_time=$(date +%s)
elapsed=$((end_time - start_time))

if [ $exit_code -eq 0 ]; then
    echo "Clover 运行完成！结果已保存到: $OUTPUT_FILE"
    echo "运行耗时: $elapsed 秒"
else
    echo "运行出错，请检查输入文件或参数"
fi
