import os
import json
import glob

def convert_arc_to_rwkv(arc_file_path, output_file_path):
    """将ARC数据集文件转换为RWKV格式的JSONL文件
    
    Args:
        arc_file_path: ARC数据集JSON文件路径
        output_file_path: 输出的JSONL文件路径
    """
    # 读取ARC数据集文件
    with open(arc_file_path, 'r', encoding='utf-8') as f:
        arc_data = json.load(f)
    
    # 创建输出文件
    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        # 处理训练数据
        if 'train' in arc_data:
            for example in arc_data['train']:
                # 将输入和输出转换为字符串格式
                input_str = str(example['input']).replace(' ', '')
                output_str = str(example['output']).replace(' ', '')
                
                # 创建RWKV格式的文本
                rwkv_text = f"<task><input>{input_str}</input><output>{output_str}</output></task>"
                
                # 写入JSONL格式
                json_line = {"text": rwkv_text}
                out_f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        
        # 处理测试数据
        if 'test' in arc_data:
            for example in arc_data['test']:
                # 将输入和输出转换为字符串格式
                input_str = str(example['input']).replace(' ', '')
                output_str = str(example['output']).replace(' ', '')
                
                # 创建RWKV格式的文本
                rwkv_text = f"<task><input>{input_str}</input><output>{output_str}</output></task>"
                
                # 写入JSONL格式
                json_line = {"text": rwkv_text}
                out_f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

def convert_arc_dataset(arc_dir, output_dir):
    """转换整个ARC数据集目录
    
    Args:
        arc_dir: ARC数据集目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(arc_dir, '*.json'))
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 处理每个文件
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        output_file = os.path.join(output_dir, file_name.replace('.json', '.jsonl'))
        
        print(f"处理文件: {file_name}")
        convert_arc_to_rwkv(json_file, output_file)
    
    print("转换完成！")

def merge_jsonl_files(input_dir, output_file, epochs=1):
    """合并多个JSONL文件为一个，并可以重复数据多个epoch
    
    Args:
        input_dir: 包含JSONL文件的目录
        output_file: 输出的合并JSONL文件
        epochs: 数据重复的次数
    """
    # 获取所有JSONL文件
    jsonl_files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件进行合并")
    
    # 读取所有行
    all_lines = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_lines.extend(lines)
    
    print(f"总共读取了 {len(all_lines)} 行数据")
    
    # 写入合并文件，重复epochs次
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in range(epochs):
            for line in all_lines:
                f.write(line)
    
    print(f"已将数据重复 {epochs} 次并合并到 {output_file}")
    print(f"最终文件包含 {len(all_lines) * epochs} 行数据")

def main():
    # 设置路径
    arc_training_dir = "c:\\code\\train_temp\\ARC-AGI-2\\data\\training"
    output_dir = "c:\\code\\train_temp\\arc_rwkv_format"
    final_output = "c:\\code\\train_temp\\arc_combined.jsonl"
    
    # 转换数据集
    convert_arc_dataset(arc_training_dir, output_dir)
    
    # 合并为一个文件，重复3个epoch
    merge_jsonl_files(output_dir, final_output, epochs=3)
    
    print(f"\n转换完成！现在可以使用makedata工具处理 {final_output} 文件")
    print("示例命令: python make_data.py arc_combined.jsonl 1 4096")

if __name__ == "__main__":
    main()