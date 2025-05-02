@echo off
echo 正在使用make_data.py处理ARC数据集...

cd %~dp0\RWKV-v7\train_temp

echo 复制ARC数据集JSONL文件到当前目录...
copy %~dp0\arc_combined.jsonl .\arc_combined.jsonl

echo 使用make_data.py处理JSONL文件...
python make_data.py arc_combined.jsonl 1 4096

echo 处理完成！
echo 生成的文件：arc_combined.bin 和 arc_combined.idx
echo 可以使用以下命令进行训练：
echo python train.py --load_model (你的模型路径) --proj_dir (项目目录) --data_file arc_combined --my_exit_tokens (token数) --magic_prime (质数) --ctx_len 4096

pause