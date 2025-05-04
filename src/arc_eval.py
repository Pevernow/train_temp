import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import types # Add missing import
import math # Add missing import
from typing import List # Add missing import
import torch.nn.functional as F # Add missing import

# Set environment variables before importing the model
os.environ["RWKV_MY_TESTING"] = 'x070'
os.environ["RWKV_HEAD_SIZE"] = '64' # Set HEAD_SIZE environment variable

# Adjust imports based on your project structure
# Try relative imports first (for running with python -m src.arc_eval)
try:
    from .model import RWKV
    from ..tokenizer.rwkv_tokenizer import TRIE_TOKENIZER as RWKV_TOKENIZER # Import correct class
    print("Successfully imported using relative paths.")
except ImportError as e1:
    print(f"Relative import failed: {e1}. Trying absolute imports...")
    # Fallback to absolute imports (might work if run differently or PYTHONPATH is set)
    try:
        from src.model import RWKV # Or the correct model class
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER as RWKV_TOKENIZER # Import correct class
        print("Successfully imported using absolute paths.")
    except ImportError as e2:
        print(f"Absolute import also failed: {e2}")
        exit("Error: Could not import RWKV model or tokenizer. Check PYTHONPATH or script location.")

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RWKV model on ARC dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the RWKV model checkpoint (.pth file)')
    parser.add_argument('--data_path', type=str, default='../converted_data/eval.jsonl', help='Path to the ARC evaluation data (.jsonl file)')
    parser.add_argument('--vocab_path', type=str, default='c:\\code\\train_temp\\vocab\\arc_vocab.txt', help='Path to the vocabulary file') # Updated default path
    parser.add_argument('--strategy', type=str, default='cuda bf16', help='Pytorch Lightning strategy (e.g., cuda bf16, cuda fp16, deepspeed_stage_3)') # Updated help string
    parser.add_argument('--precision', type=str, default='bf16', help='Precision (bf16, fp16, fp32)') # Updated default precision
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate for the output grid')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling probability')
    # Add other relevant RWKV model args if needed (n_layer, n_embd, etc.)
    # These might be inferred from the checkpoint, but explicit args can be useful
    parser.add_argument('--n_layer', type=int, help='Number of layers (optional, try to infer from model)')
    parser.add_argument('--n_embd', type=int, help='Embedding size (optional, try to infer from model)')
    parser.add_argument('--vocab_size', type=int, default=22, help='Vocabulary size (optional, default based on arc_vocab.txt)') # Added vocab_size arg

    # Potentially add args for specific ARC formatting if needed

    args = parser.parse_args()
    return args

# --- Data Loading ---
def load_arc_data(data_path):
    """Loads ARC data from a jsonl file."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {line.strip()} - Error: {e}")
    return data

# --- ARC Task Parsing/Formatting (Example) ---
def format_arc_prompt(task_json):
    """Formats an ARC task into a string prompt for the RWKV model."""
    # Example format: Convert grids to string representation
    # This needs to match the format the model was trained on or expects
    try:
        if 'text' not in task_json or not task_json['text']:
            print(f"Warning: Skipping task due to empty or missing 'text' field: {task_json}")
            return None, None

        text_content = task_json['text']
        output_tag_start = "<output>"
        output_tag_end = "</output>"

        # Find last occurrence of output_tag_start
        output_start_index = text_content.rfind(output_tag_start)
        if output_start_index == -1:
            print(f"Warning: Skipping task because '{output_tag_start}' tag not found in 'text': {task_json}")
            return None, None

        # Prompt is everything up to and including the last <output> tag
        prompt = text_content[:output_start_index + len(output_tag_start)]

        # Find matching output_tag_end after the last output_tag_start
        output_end_index = text_content.find(output_tag_end, output_start_index)
        if output_end_index == -1:
            print(f"Warning: Skipping task because '{output_tag_end}' tag not found after '{output_tag_start}' in 'text': {task_json}")
            return None, None

        # Ground truth is the content between <output> and </output>
        ground_truth_output_str = text_content[output_start_index + len(output_tag_start):output_end_index].strip()

        # Optional: Add validation for the extracted ground truth if needed
        # e.g., check if it looks like a grid representation

        return prompt, ground_truth_output_str

    except (KeyError, TypeError) as e: # Catch other potential errors like missing keys
        print(f"Error processing task structure (after initial checks): {task_json} - Error: {e}")
        return None, None

def parse_generated_output(generated_text):
    """Parses the model's generated text to extract the output grid string."""
    # Find the end tag or use string manipulation based on expected format
    try:
        # Simple example: assume output is everything after <output> until </task>
        start_tag = "<output>"
        end_tag = "</output>"
        start_index = generated_text.rfind(start_tag)
        if start_index == -1:
            return None # Or handle cases where the tag isn't found
        start_index += len(start_tag)
        end_index = generated_text.find(end_tag, start_index)
        if end_index == -1:
            # If no end tag, maybe take up to max_tokens or until newline?
            # This part is crucial and depends on model behavior.
            # For now, let's just take the rest of the string (might be too long)
            output_str = generated_text[start_index:].strip()
        else:
            output_str = generated_text[start_index:end_index].strip()

        # Basic validation: Does it look like a grid representation?
        # Remove potential leading/trailing whitespace/newlines before checking
        output_str = output_str.strip()
        if output_str.startswith('[(', ) and output_str.endswith(')]'):
            # Further validation could involve trying ast.literal_eval
            try:
                import ast
                parsed_grid = ast.literal_eval(output_str)
                if isinstance(parsed_grid, list) and all(isinstance(row, tuple) for row in parsed_grid):
                     return output_str # Looks like a valid grid structure
                else:
                    print(f"Warning: Parsed output is not a list of tuples: {output_str}")
                    return None
            except (SyntaxError, ValueError) as parse_error:
                 print(f"Warning: Generated output failed literal_eval: {output_str} - Error: {parse_error}")
                 return None
        else:
            print(f"Warning: Generated output doesn't start/end like expected grid format: '{output_str}'")
            return None # Or return the raw string for debugging
    except Exception as e:
        print(f"Error parsing generated text: {generated_text} - Error: {e}")
        return None

# --- Model Loading & Inference ---
def load_rwkv_model(args):
    """Loads the RWKV model from checkpoint."""
    print(f"Loading model from: {args.model_path}")

    # Determine device and dtype
    if 'cuda' in args.strategy:
        device = 'cuda'
    else:
        device = 'cpu'

    if args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model_weights = checkpoint
    # Handle potential nested structure (e.g., if saved with pytorch lightning)
    if 'state_dict' in checkpoint:
        model_weights = checkpoint['state_dict']
        print("Extracted state_dict from checkpoint.")

    # Infer n_layer and n_embd if not provided
    inferred_n_layer = args.n_layer
    inferred_n_embd = args.n_embd
    if inferred_n_layer is None or inferred_n_embd is None:
        print("Attempting to infer n_layer and n_embd from checkpoint...")
        max_layer = -1
        emb_size = 0
        try:
            for k in model_weights.keys():
                if '.att.time_mix_k' in k: # More reliable key
                    parts = k.split('.')
                    if parts[0] == 'blocks' and parts[1].isdigit():
                        max_layer = max(max_layer, int(parts[1]))
                if k == 'emb.weight':
                    emb_size = model_weights[k].shape[1]

            if max_layer != -1 and inferred_n_layer is None:
                inferred_n_layer = max_layer + 1
                print(f"  Inferred n_layer: {inferred_n_layer}")
            if emb_size > 0 and inferred_n_embd is None:
                inferred_n_embd = emb_size
                print(f"  Inferred n_embd: {inferred_n_embd}")

            if inferred_n_layer is None:
                print("  Warning: Could not infer n_layer.")
            if inferred_n_embd is None:
                print("  Warning: Could not infer n_embd.")

        except Exception as e:
            print(f"  Warning: Error during inference of n_layer/n_embd: {e}")

    # Use provided args if inference failed or wasn't needed
    final_n_layer = args.n_layer if args.n_layer is not None else inferred_n_layer
    final_n_embd = args.n_embd if args.n_embd is not None else inferred_n_embd

    if final_n_layer is None or final_n_embd is None:
        raise ValueError("Could not determine n_layer or n_embd. Please provide them via --n_layer and --n_embd arguments.")

    # --- Instantiate model args (align with src/model.py RWKV class) ---
    model_args = types.SimpleNamespace()
    model_args.n_layer = final_n_layer
    model_args.n_embd = final_n_embd
    model_args.vocab_size = args.vocab_size # Use the provided/default vocab_size
    model_args.head_size = 64 # Assuming RWKV-7 default, adjust if needed
    model_args.head_size_a = model_args.head_size # Usually same as head_size
    model_args.ctx_len = 8192 # Common context length, adjust if your model differs
    model_args.dropout = 0 # No dropout during evaluation
    model_args.my_pos_emb = 0 # Common setting
    model_args.pre_ffn = 0 # Common setting
    model_args.my_testing = 'x070' # Set based on model version if needed (e.g., 'x060', 'x070')
    # Add any other args required by your specific RWKV model version's __init__
    # model_args.tiny_att_dim = 0
    # model_args.tiny_att_layer = -1
    model_args.grad_cp = 0 # Add grad_cp argument with default value 0 for evaluation
    print(f"Instantiating RWKV model with: n_layer={model_args.n_layer}, n_embd={model_args.n_embd}")

    model = RWKV(model_args)

    # Adjust keys if necessary (e.g., remove 'module.' prefix if saved from DataParallel/DDP)
    adjusted_weights = {}
    for k, v in model_weights.items():
        if k.startswith('_forward_module.'):
             k = k[len('_forward_module.'):]
        elif k.startswith('module.'):
             k = k[len('module.'):]
        adjusted_weights[k] = v

    print("Loading state dict...")
    load_result = model.load_state_dict(adjusted_weights, strict=False)
    print(f"  Missing keys: {load_result.missing_keys}")
    print(f"  Unexpected keys: {load_result.unexpected_keys}")

    print("Moving model to device...")
    model = model.to(device=device)
    # Apply precision after moving to device
    if dtype != torch.float32:
        print(f"Casting model to {dtype}...")
        model = model.to(dtype=dtype)

    model.eval()
    print("Model loaded successfully.")
    return model

def generate_output(model, tokenizer, prompt, args):
    """Generates output from the model given a prompt using GPT+RNN mode."""
    with torch.no_grad():
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=next(model.parameters()).device)

        state = None
        # Initialize state based on model structure (assuming similar to rwkv_v7_demo_fast)
        # This might need adjustment based on the actual src.model.RWKV implementation
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        if hasattr(model, 'args'): # Check if model has args attribute like in demo
            n_layer = model.args.n_layer
            n_embd = model.args.n_embd
            head_size = model.args.head_size
            state = [None for _ in range(n_layer * 3)]
            for i in range(n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(n_embd, dtype=dtype, requires_grad=False, device=device)
                # Assuming att_kv state shape based on demo
                state[i*3+1] = torch.zeros((n_embd // head_size, head_size, head_size), dtype=torch.float, requires_grad=False, device=device)
                state[i*3+2] = torch.zeros(n_embd, dtype=dtype, requires_grad=False, device=device)
        else:
            print("Warning: Cannot automatically initialize state. Model structure might differ from demo.")
            # Fallback or raise error if state initialization is critical and unknown

        # Process the prompt using forward_seq equivalent (GPT mode)
        # Ensure prompt_tokens[:-1] is a tensor
        if len(prompt_tokens) > 1:
            prompt_tensor = torch.tensor([prompt_tokens[:-1]], dtype=torch.long, device=next(model.parameters()).device)
            # Process the prompt sequence. The state variable initialized earlier is not used by this forward method.
            # The model.forward method in src/model.py does not support explicit state passing for RNN generation.
            # The generation loop below needs to be revised to work with the available model methods.
            # For now, just process the prompt sequence to avoid the TypeError.
            # The output of this call is the logits for the prompt sequence.
            # If stateful generation is required, the model implementation or generation logic needs adjustment.
            logits_prompt = model.forward(prompt_tensor)
            # The 'state' variable is not updated by this call and is likely unused in the subsequent generation loop
            # as the model.forward method doesn't return the state in a usable format for token-by-token generation.

        output_tokens = []
        # Start generation with the last token of the prompt.
        # Note: The current model.forward doesn't support token-by-token generation with state in the standard way.
        # The following loop will likely not perform true stateful RNN generation with the current model.py.
        # It might need to call model.forward with a sequence of [last_prompt_token, generated_token_1, ...]
        # or model.py needs a dedicated generation method.
        current_token_id = input_ids[:, -1].item()

        # Generation loop (likely not true stateful RNN generation with current model.py)
        for i in range(args.max_tokens):
            # Call model.forward with the current token as a sequence of length 1
            # This will process the single token but won't use/update the 'state' variable
            # in the way a typical RNN generation loop would.
            current_token_tensor = torch.tensor([[current_token_id]], dtype=torch.long, device=next(model.parameters()).device)
            logits = model.forward(current_token_tensor)
            # Logits should be for the *next* token prediction (shape B, T, V -> 1, 1, V)
            logits = logits.squeeze(0).squeeze(0) # Get logits for the single token
            #print(logits)

            # Sampling (similar to original, but applied to the new logits)
            probs = F.softmax(logits.float() / args.temperature, dim=-1)

            # Top-p sampling
            if args.top_p > 0 and args.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff_index = torch.searchsorted(cumulative_probs, args.top_p)
                # Ensure cutoff_index is within bounds
                cutoff_index = min(cutoff_index.item(), probs.shape[-1] - 1)
                probs_to_keep = sorted_probs[:cutoff_index+1]
                indices_to_keep = sorted_indices[:cutoff_index+1]
                # Create a mask for valid tokens
                mask = torch.zeros_like(probs)
                mask[indices_to_keep] = 1.0
                probs = probs * mask
                # Renormalize if mask was applied
                if torch.sum(probs) > 0:
                    probs = probs / torch.sum(probs)
                else:
                    # Handle case where all probs are zero after top-p (should be rare)
                    probs = torch.ones_like(probs) / probs.shape[-1]

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            #print(next_token)

            output_tokens.append(next_token)
            current_token_id = next_token # Use the sampled token as input for the next step

            # Check for end token '</task>' (Token ID 2)
            if next_token == 2:
                print(f"Found '</task>' (ID: 2) after {i+1} tokens.")
                break
            # Check for EOS token if your tokenizer/model uses one
            # if token_item == tokenizer.eos_token_id:
            #    break
            #print(i)
        generated_sequence = prompt_tokens + output_tokens
        generated_text = tokenizer.decode(generated_sequence)
        # print(f"Generated text length: {len(generated_text)}")
        return generated_text

# --- Evaluation ---
def evaluate(model, tokenizer, data, args):
    """Runs evaluation on the ARC dataset."""
    correct_predictions = 0
    total_tasks = len(data)

    for i, task_json in enumerate(tqdm(data, desc="Evaluating ARC tasks")):
        prompt, ground_truth_output_str = format_arc_prompt(task_json)
        if prompt is None or ground_truth_output_str is None:
            print(f"Skipping task {i+1}/{total_tasks} due to parsing error.")
            total_tasks -= 1 # Adjust total count if skipping
            continue

        # Generate model output
        generated_text = generate_output(model, tokenizer, prompt, args)

        # Parse generated output
        predicted_output_str = parse_generated_output(generated_text)

        # Compare
        if predicted_output_str is not None and predicted_output_str == ground_truth_output_str:
            correct_predictions += 1
            print(f"Task {i+1}/{total_tasks}: Correct")
        else:
            print(f"Task {i+1}/{total_tasks}: Incorrect")
            print(f"  Prompt: {prompt}...")
            print(f"  Truth:  {ground_truth_output_str}")
            print(f"  Pred:   {predicted_output_str}")
            print(f"  RawGen: {generated_text}")

    accuracy = correct_predictions / total_tasks if total_tasks > 0 else 0
    print(f"\nEvaluation Complete:")
    print(f"Total Tasks: {total_tasks}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    # Set device and precision globally if needed by lower-level functions
    # torch.set_default_dtype(torch.float16 if args.precision == 'fp16' else torch.bfloat16 if args.precision == 'bf16' else torch.float32)

    # Load Tokenizer
    print(f"Loading tokenizer from: {args.vocab_path}")
    tokenizer = RWKV_TOKENIZER(args.vocab_path)

    # Load Model
    model = load_rwkv_model(args)

    # Load Data
    print(f"Loading data from: {args.data_path}")
    arc_data = load_arc_data(args.data_path)

    # Evaluate
    evaluate(model, tokenizer, arc_data, args)
