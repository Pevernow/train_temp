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
    遍历arc_dir下所有json文件，将每个文件的所有train样本合并到train.jsonl的一行，
    将每个文件的所有test样本合并到test.jsonl的一行。
    每行格式为 {"text": "<task><input>...</input><output>...</output><input>...</input><output>...</output>...</task>"}。
    """
    os.makedirs(output_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(arc_dir, '*.json'))
    print(f"找到 {len(json_files)} 个JSON文件")

    train_output_path = os.path.join(output_dir, 'train.jsonl')
    test_output_path = os.path.join(output_dir, 'test.jsonl')

    # 清空或创建文件
    with open(train_output_path, 'w', encoding='utf-8') as f: pass
    with open(test_output_path, 'w', encoding='utf-8') as f: pass

    total_train_files = 0
    total_test_files = 0

    for json_file in json_files:
        # print(f"处理文件: {os.path.basename(json_file)}") # 可选：打印正在处理的文件名
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                arc_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"错误：无法解析JSON文件 {json_file}: {e}")
                continue # 跳过这个文件

        # 处理train
        train_task_content = ""
        if 'train' in arc_data and arc_data['train']:
            for example in arc_data['train']:
                if 'input' not in example or 'output' not in example:
                    print(f"警告：跳过 {json_file} 中缺少 'input' 或 'output' 的训练样本")
                    continue
                # 交换 input 和 output
                output_as_input_str = convert_brackets_to_square_brackets(example['output']) # 输出作为输入，用方括号
                input_as_output_str = convert_brackets_to_square_brackets(example['input'])   # 输入作为输出，用方括号
                train_task_content += f'<input>{output_as_input_str}</input><output>{input_as_output_str}</output>'
            if train_task_content:
                task_str = f'<task>{train_task_content}</task>'
                json_line = json.dumps({"text": task_str}, ensure_ascii=False)
                with open(train_output_path, 'a', encoding='utf-8') as f_train:
                    f_train.write(json_line + '\n')
                total_train_files += 1

        # 处理test
        test_task_content = ""
        if 'test' in arc_data and arc_data['test']:
            for example in arc_data['test']:
                if 'input' not in example:
                    print(f"警告：跳过 {json_file} 中缺少 'input' 的测试样本")
                    continue
                # 交换 input 和 output
                output_as_input_str = "" # 默认为空
                if 'output' in example:
                    output_as_input_str = convert_brackets_to_square_brackets(example['output']) # 输出作为输入，用方括号
                input_as_output_str = convert_brackets_to_square_brackets(example['input'])   # 输入作为输出，用方括号

                # 始终包含input和output标签，即使output为空
                test_task_content += f'<input>{output_as_input_str}</input><output>{input_as_output_str}</output>'

            if test_task_content:
                task_str = f'<task>{test_task_content}</task>'
                json_line = json.dumps({"text": task_str}, ensure_ascii=False)
                with open(test_output_path, 'a', encoding='utf-8') as f_test:
                    f_test.write(json_line + '\n')
                total_test_files += 1

    print(f"已处理 {len(json_files)} 个文件，生成 {total_train_files} 行训练数据到 {train_output_path}")
    print(f"已处理 {len(json_files)} 个文件，生成 {total_test_files} 行测试数据到 {test_output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='将ARC数据集转换为JSONL格式，每个文件一行')
    parser.add_argument('--arc_dir', type=str, required=True, help='ARC数据集目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    args = parser.parse_args()
    convert_arc_dataset_to_jsonl_per_file(args.arc_dir, args.output_dir)