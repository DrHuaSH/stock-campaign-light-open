#!/bin/bash
# 快速安装依赖脚本（使用本地缓存）

echo "正在从本地 packages 目录安装依赖..."
./venv/bin/pip install --no-index --find-links=packages -r requirements.txt

echo "安装完成！"
