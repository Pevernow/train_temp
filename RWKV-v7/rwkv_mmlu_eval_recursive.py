########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os, json, datetime, random
from tqdm import tqdm
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
from datasets import load_dataset, load_from_disk

# Import custom RWKV implementation and tokenizer
from rwkv_v7_demo import RWKV, RWKV_TOKENIZER, args as model_args # Import args as well

########################################################################################################

# Use user's model path
MODEL_NAME = "../rwkv-3.pth"
print(f"Loading model - {MODEL_NAME}")

# Configure args for the custom model (adjust based on rwkv_v7_demo.py if needed)
# These might be automatically handled if rwkv_v7_demo.py sets them globally,
# but explicitly setting them here based on the demo script is safer.
model_args.n_layer = 12 # Example, adjust if your model differs
model_args.n_embd = 768 # Example, adjust if your model differs
model_args.vocab_size = 65536
model_args.head_size_a = 64
# model_args.DTYPE = torch.bfloat16 # Or torch.half based on your demo script
model_args.DTYPE = torch.half

# Load the custom model
model = RWKV(model_args)
model_weights = torch.load(MODEL_NAME, map_location='cpu')
model_weights = {k.replace('_orig_mod.', '') if '_orig_mod.' in k else k: v for k, v in model_weights.items()} # Adjust key names if needed
model.load_state_dict(model_weights, strict=False) # Use strict=False initially
model = model.to(model_args.DTYPE).cuda().eval()

# Use the custom tokenizer
tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")

RECURSIVE_DEPTH = 10 # Set the desired recursive depth
print(f"Using recursive depth: {RECURSIVE_DEPTH}")

########################################################################################################

mmlu_test = load_from_disk("mmlu_test_dataset")
mmlu_dev = load_from_disk("mmlu_dev_dataset")

TEMPLATE = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is'''

CHOICES = [" A", " B", " C", " D"]

SHUFFLE = False
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################

correct = 0
total = 0
pbar = tqdm(total=len(mmlu_test))

choices_token = [tokenizer.encode(x) for x in CHOICES]
assert all([len(x) == 1 for x in choices_token]), "Choices are not single token, use rwkv_mmlu.py instead"
choices_token = [x[0] for x in choices_token]

for idx, sample in enumerate(mmlu_test):
    question = sample["question"]
    choices = sample["choices"]
    subject = sample["subject"]
    gt = sample["answer"]

    if SHUFFLE and not any(["Both" in x for x in choices]):  # exclude choices like "Both A and B"
        original_gt_text = choices[gt]
        np.random.shuffle(choices)
        gt = choices.index(original_gt_text)

    all_prefix = (
        TEMPLATE.replace("<Q>", question)
        .replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
        .replace("<SUBJECT>", subject.replace("_", " "))
    )

    if idx == 0:
        print(f"Format example:")
        print("-" * 100)
        print(all_prefix)
        print("-" * 100)
        format_example = all_prefix

    all_prefix_ids = [0] + tokenizer.encode(all_prefix.replace('\r\n','\n').strip())

    # Use the custom model's forward method with recursive_depth
    # The custom forward likely expects a tensor input
    input_tensor = torch.tensor([all_prefix_ids], dtype=torch.long).cuda()
    logits = model.forward(input_tensor, recursive_depth=RECURSIVE_DEPTH)
    # Get the logits for the last token
    logits = logits[0, -1, :]

    neg_log_prob = F.log_softmax(logits.float(), dim=-1) # Ensure float for softmax
    target_prob = neg_log_prob[choices_token]
    
    if torch.argmax(target_prob).item() == gt:
        correct += 1
    total += 1
    pbar.set_description(f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")
    pbar.update(1)
pbar.close()