import os
import json
import glob

def convert_brackets_to_square_brackets(data):
    """
    将嵌套list转换为方括号格式
    例如：[[1,2],[3,4]] -> [(1,2),(3,4)]
    """
    if isinstance(data, list):
        inner = ','.join([convert_brackets_to_square_brackets(item) for item in data])
        # 注意：这里返回的是带方括号的字符串，内部仍然是小括号
        # 如果内部也需要方括号，需要递归修改
        # 根据用户示例 ((7,9),(4,3)) -> [(7,9),(4,3)]，内部保持小括号
        def convert_inner_to_parentheses(inner_data):
            if isinstance(inner_data, list):
                inner_str = ','.join([convert_inner_to_parentheses(i) for i in inner_data])
                return f"({inner_str})"
            else:
                return str(inner_data)
        inner_paren = ','.join([convert_inner_to_parentheses(item) for item in data])
        return f"[{inner_paren}]"
    else:
        return str(data)

def convert_arc_dataset_to_jsonl_per_file(arc_dir, output_dir):
    """
    遍历arc_dir下所有json文件，将每个文件的train样本追加到train.jsonl，test样本追加到eval.jsonl。
    每个JSON文件中的每个样本对应输出文件中的一行。
    每行格式为 {"text": "<task><input>...</input><output>...</output></task>"}。
    原始input对应<input>标签，原始output对应<output>标签。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 分别获取training和evaluation文件夹中的JSON文件
    train_files = glob.glob(os.path.join(arc_dir, 'training', '*.json'))
    eval_files = glob.glob(os.path.join(arc_dir, 'evaluation', '*.json'))
    print(f"找到 {len(train_files)} 个训练JSON文件和 {len(eval_files)} 个评估JSON文件")

    train_output_path = os.path.join(output_dir, 'train.jsonl')
    eval_output_path = os.path.join(output_dir, 'eval.jsonl')

    # 清空或创建文件
    with open(train_output_path, 'w', encoding='utf-8') as f_train: pass
    with open(eval_output_path, 'w', encoding='utf-8') as f_eval: pass

    total_train_samples = 0
    total_eval_samples = 0

    # 处理训练集文件
    for train_file in train_files:
        with open(train_file, 'r', encoding='utf-8') as f:
            try:
                arc_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"错误：无法解析训练文件 {train_file}: {e}")
                continue

        # 收集当前文件的所有样本
            task_contents = []
            for part in ['train', 'test']:
                if part in arc_data and arc_data[part]:
                    for example in arc_data[part]:
                        if 'input' not in example:
                            print(f"警告：跳过 {train_file} 中缺少 'input' 的{part}样本")
                            continue
                        input_str = convert_brackets_to_square_brackets(example['input'])
                        output_str = convert_brackets_to_square_brackets(example.get('output', ''))
                        task_contents.append(f'<input>{input_str}</input><output>{output_str}</output>')
            
            # 合并所有样本为单行输出
            if task_contents:
                combined_tasks = ''.join(task_contents)
                json_line = json.dumps({"text": f'<task>{combined_tasks}</task>'}, ensure_ascii=False)
                with open(train_output_path, 'a', encoding='utf-8') as f_train:
                    f_train.write(json_line + '\n')
                total_train_samples += 1  # 每文件计为1行

    # 处理评估集文件（修改后）
    for eval_file in eval_files:
        with open(eval_file, 'r', encoding='utf-8') as f:
            try:
                arc_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"错误：无法解析评估文件 {eval_file}: {e}")
                continue
    
        # 处理评估文件中的test部分
        combined_tasks = []
        if 'test' in arc_data and arc_data['test']:
            print(f"正在处理评估文件 {eval_file}，发现 {len(arc_data['test'])} 个测试样本")
            task_contents = []
            for example in arc_data['test']:
                if 'input' not in example:
                    print(f"警告：跳过 {eval_file} 中缺少 'input' 的test样本")
                    continue
                input_str = convert_brackets_to_square_brackets(example['input'])
                output_str = convert_brackets_to_square_brackets(example.get('output', ''))
                task_contents.append(f'<input>{input_str}</input><output>{output_str}</output>')
            
            if task_contents:
                combined_tasks = f'<task>{"".join(task_contents)}</task>'
        
        if combined_tasks:
            json_line = json.dumps({"text": combined_tasks}, ensure_ascii=False)
            with open(eval_output_path, 'a', encoding='utf-8') as f_eval:
                f_eval.write(json_line + '\n')
            total_eval_samples += 1  # 按文件数统计
            print(f"成功写入评估文件到 {eval_output_path}")

    print(f"\n最终统计结果：")
    print(f"训练集: 处理了 {len(train_files)} 个文件，生成 {total_train_samples} 行数据")
    print(f"评估集: 处理了 {len(eval_files)} 个文件，生成 {total_eval_samples} 行数据")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='将ARC数据集的train部分转换为train.jsonl，test部分转换为eval.jsonl。')
    parser.add_argument('--arc_dir', type=str, required=True, help='ARC数据集目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    args = parser.parse_args()
    convert_arc_dataset_to_jsonl_per_file(args.arc_dir, args.output_dir)