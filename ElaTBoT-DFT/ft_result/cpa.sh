#!/bin/bash

# 检查是否提供了参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dir> <target_dir>"
    exit 1
fi

# 获取命令行参数
source_dir="$1"
target_dir="$2"

# 复制文件
find "$source_dir" -maxdepth 1 -type f ! -name '.*' -exec cp -r '{}' "$target_dir" \;

echo "Files copied from $source_dir to $target_dir"