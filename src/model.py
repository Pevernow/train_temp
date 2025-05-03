########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob

MyModule = nn.Module
# MyFunction = __nop # Keep functions as plain Python functions
# OR explicitly:
def MyFunction(fn): return fn # Decorator does nothing
# if os.environ["RWKV_JIT_ON"] == "1": # Comment out or remove JIT logic
#     MyModule = torch.jit.ScriptModule
#     MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

try:
    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])
except KeyError:
    print("RWKV_HEAD_SIZE environment variable not set, using default value 64.")
    HEAD_SIZE = 64 # Default value if environment variable is not set

# --- Rest of your CUDA kernel code ---
# --- (Assuming it's correct and loaded properly) ---
if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16

    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "--extra-device-vectorization"] # Removed -Xptxas -O3
    load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)


    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b): # Python Function inputs (from .apply call)
            # Input shapes: w, q, k, v, a, b typically (B, T, H, C) where C=HEAD_SIZE
            B, T, H, C = w.shape
            # --- Allocate output tensors BEFORE calling C++ op ---
            # Assuming y has the same shape as q, k, v. Adjust if different.
            y = torch.empty_like(q)
            # Determine correct shapes for s and sa based on CUDA kernel logic
            # s shape seems to be (B, H, T//CHUNK_LEN, C, C) - VERIFY THIS!
            s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
            # sa shape seems to be (B, T, H, C) - VERIFY THIS!
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            # --- Allocation done ---

            # Ensure contiguous inputs (outputs are already contiguous from empty_like/empty)
            # Note: Inputs to .apply() should ideally be made contiguous *before* calling apply if needed.
            # Adding checks here just in case.
            assert all(i.is_contiguous() for i in [w, q, k, v, a, b])
            # Removed dtype assertion for flexibility, ensure correct types are passed in practice
            # assert w.dtype == torch.bfloat16

            # Call C++ op: Pass Python forward inputs (w,q,k,v,a,b) and output buffers (y,s,sa)
            # The C++ wrapper will internally map Python 'a' to C++ 'z', Python 'b' to C++ 'a'
            torch.ops.wind_backstepping.forward(w, q, k, v, a, b, y, s, sa)

            # Save tensors needed for backward.
            # CRITICAL: Save the ORIGINAL FORWARD INPUTS (a, b) if the backward kernel needs them.
            # Also save intermediate results s, sa if the backward kernel uses them.
            # Also save tensors whose gradients are needed (w, q, k, v).
            ctx.save_for_backward(w, q, k, v, a, b, s, sa) # Saving a, b, s, sa

            return y # Return the computed output y

        @staticmethod
        def backward(ctx, dy):
            assert dy.is_contiguous()
            # Retrieve saved tensors. Note: order matters if kernel uses them.
            # Assuming kernel uses w, q, k, v, a, b, s, sa
            w, q, k, v, a, b, s, sa = ctx.saved_tensors

            # Allocate gradients for forward inputs: w, q, k, v, a, b
            dw = torch.empty_like(w)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            da_grad = torch.empty_like(a) # Gradient for forward input 'a' (which was -kk)
            db_grad = torch.empty_like(b) # Gradient for forward input 'b' (which was kk*a)

            # Call C++ backward. It expects outputs dz, da (corresponding to C++ inputs z, a)
            # Pass memory locations da_grad and db_grad to receive these results.
            torch.ops.wind_backstepping.backward(w, q, k, v, a, b, dy, s, sa, dw, dq, dk, dv, da_grad, db_grad)
            #                                                                          ^        ^
            #                                                               C++ writes dz here | C++ writes da here

            # Return gradients corresponding to forward inputs w, q, k, v, a, b
            return dw, dq, dk, dv, da_grad, db_grad # Return gradients for a and b

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b, head_size: int): # Add head_size parameter
        B, T, HC = q.shape
        # Use the passed argument instead of the global variable
        H = HC // head_size
        # Calculate C using the passed argument as well
        C = head_size
        q, w, k, v, a, b = [i.view(B, T, H, C).contiguous() for i in [q, w, k, v, a, b]] # Ensure contiguous views

        # Assuming the parameter mapping a->z, b->b is correct for WindBackstepping
        # ****** Double-check this mapping based on your CUDA kernel's actual inputs ******
        # Ensure inputs have the correct dtype expected by the kernel (e.g., bfloat16)
        # Add type casting if necessary, e.g., w.to(torch.bfloat16)
        expected_dtype = torch.bfloat16 # Or float32, check kernel
        w, q, k, v, a, b = [i.to(expected_dtype) for i in [w, q, k, v, a, b]]

        output = WindBackstepping.apply(w, q, k, v, a, b)
        return output.view(B, T, HC) # Return result reshaped

########################################################################################################

# --- RWKV_Tmix_x070 class (keep as is) ---
class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size # Use self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a, self.head_size) # Pass head_size here


        # Make sure C used here is consistent (it comes from x.size() initially)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # Original code used H and implicitly HEAD_SIZE/N here. Ensure consistency.
        # N = self.head_size # Should be consistent
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

# --- RWKV_CMix_x070 class (keep as is) ---
class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    # @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

# --- Block class (keep as is) ---
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        # Adjust block logic based on pre_ffn (should already be correct)
        if self.layer_id == 0 and self.args.pre_ffn > 0:
             self.ffnPre = RWKV_CMix_x070(args, 0)
             self.att = RWKV_Tmix_x070(args, layer_id) # Time mix happens after pre_ffn in layer 0
        else:
             self.att = RWKV_Tmix_x070(args, layer_id) # Normal order

        self.ffn = RWKV_CMix_x070(args, layer_id)

        # Tiny Attention logic (seems okay, but ensure RWKV_Tmix_x070 handles head_size correctly)
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            # Ensure tiny_att uses appropriate head size if different
            # Need to pass tiny_head_size to RWKV_Tmix_x070 if it relies on args only
            # Assuming RWKV_Tmix_x070 uses args.head_size, this might need adjustment
            # Or modify RWKV_Tmix_x070 to accept head_size override
            print(f"WARNING: Tiny ATT layer {args.tiny_att_layer} uses main head_size ({args.head_size}) for RWKV_Tmix_x070. Ensure this is intended.")
            self.tiny_att = RWKV_Tmix_x070(args, 0)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    def forward(self, x, v_first):
        args = self.args
        if self.layer_id == 0:
            x = self.ln0(x)
            # Positional embedding logic (seems okay)
            if args.my_pos_emb > 0:
                 pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(args.my_pos_emb ** 2, -1)
                 x = x + pos_emb[:x.shape[1],:]

        # Apply pre_ffn if applicable
        if self.layer_id == 0 and args.pre_ffn > 0:
             x = x + self.ffnPre(self.ln1(x)) # Apply pre_ffn before attention
             x_attn, v_first = self.att(self.ln2(x), v_first) # Use ln2 for attention input
             x = x + x_attn # Add attention output
        else:
             # Normal block structure
             x_attn, v_first = self.att(self.ln1(x), v_first)
             x = x + x_attn
             x = x + self.ffn(self.ln2(x))

        # Tiny Attention Layer Logic
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
             # Assuming tiny_att needs ln before it
             x = x + self.tiny_att(self.tiny_ln(x), v_first)[0] # Only need output, v_first state from main att is fine? Check logic.

        # Dropout application (seems okay)
        if args.dropout > 0:
            x = self.drop1(x)

        return x, v_first

# --- L2Wrap class (keep as is) ---
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'): args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'): args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'): args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'): args.tiny_att_dim = -1
        # Ensure tiny_head_size exists, default to head_size if not
        if not hasattr(args, 'tiny_head_size'): args.tiny_head_size = getattr(args, 'head_size', 64) # Default head_size too
        if not hasattr(args, 'dropout'): args.dropout = 0
        if not hasattr(args, 'head_size_divisor'): args.head_size_divisor = 8
        if not hasattr(args, 'my_pos_emb'): args.my_pos_emb = 0
        if not hasattr(args, 'pre_ffn'): args.pre_ffn = 0

        # Add head_size default if missing in args (needed by Tmix)
        if not hasattr(args, 'head_size'): args.head_size = 64
        args.dim_att = args.n_embd # Ensure dim_att is n_embd for Tmix head calc if not set

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # Store token IDs for masking
        # These should ideally come from the tokenizer/vocab used for data prep
        # Using values from your arc_vocab.txt
        self.output_start_token_id = 5
        self.output_end_token_id = 6
        self.task_end_token_id = 2
        self.ignore_index = -100 # Standard ignore index for CrossEntropyLoss

    # --- configure_optimizers (keep as is) ---
    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        # Correctly check for parameters requiring grad before adding to optimizer groups
        for n, p in self.named_parameters():
            if not p.requires_grad:
                 continue # Skip parameters that don't require gradients

            is_decay = False
            is_2x = False

            # Check for 2x learning rate (e.g., specific attention params)
            # Adjust conditions based on RWKV v7 parameter names if needed
            if ("att.w0" in n): # Example, verify for v7
                 is_2x = True

            # Check for weight decay application (typically dense layers' weights)
            if len(p.squeeze().shape) >= 2 and args.weight_decay > 0 and n.endswith(".weight"):
                 # Add exclusions if necessary (e.g., norms, embeddings)
                 if not any(exclude in n for exclude in ['ln', 'norm', 'emb']):
                     is_decay = True

            # Assign to appropriate group
            if is_2x:
                 lr_2x.add(n)
            elif is_decay:
                 lr_decay.add(n)
            else:
                 lr_1x.add(n)

        # Sort for consistency
        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print('Optimizing with settings:')
            print(f'  Learning Rate Init: {args.lr_init}')
            print(f'  Betas: {args.betas}')
            print(f'  Adam Epsilon: {args.adam_eps}')
            print(f'  Weight Decay: {args.weight_decay}')
            print(f"  Parameters with WD: {len(lr_decay)}")
            # print('  WD Param Names:', lr_decay) # Optional: very verbose
            print(f"  Parameters with 1x LR: {len(lr_1x)}")
            # print('  1x Param Names:', lr_1x) # Optional: very verbose
            print(f"  Parameters with 2x LR: {len(lr_2x)}")
            # print('  2x Param Names:', lr_2x) # Optional: very verbose


        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}

        # Ensure sets only contain parameters present in param_dict
        lr_1x_params = [param_dict[n] for n in lr_1x if n in param_dict]
        lr_2x_params = [param_dict[n] for n in lr_2x if n in param_dict]
        lr_decay_params = [param_dict[n] for n in lr_decay if n in param_dict]


        optim_groups = []
        if lr_1x_params:
            optim_groups.append({"params": lr_1x_params, "weight_decay": 0.0, "my_lr_scale": 1.0})
        if lr_2x_params:
             optim_groups.append({"params": lr_2x_params, "weight_decay": 0.0, "my_lr_scale": 2.0})
        if lr_decay_params and args.weight_decay > 0:
             optim_groups.append({"params": lr_decay_params, "weight_decay": args.weight_decay, "my_lr_scale": 1.0})


        # Ensure we have parameters to optimize
        if not optim_groups:
            raise ValueError("No parameters requiring gradients found for the optimizer.")

        # Choose optimizer based on deepspeed offload status
        if self.deepspeed_offload:
            print("Using DeepSpeed CPUAdam optimizer")
            # Check if AdamW mode is appropriate based on whether weight decay is applied externally
            adamw_mode = args.weight_decay > 0
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=adamw_mode, amsgrad=False)
        else:
            print("Using FusedAdam optimizer")
             # FusedAdam handles weight decay internally via adam_w_mode
            adam_w_mode = args.weight_decay > 0
            # If we manually applied WD in optim_groups, set adam_w_mode=False? Check FusedAdam docs.
            # Typically, if WD is in optim_groups, set adam_w_mode=False. If WD is global, set adam_w_mode=True.
            # Let's assume optim_groups handles WD:
            adam_w_mode = False # Since WD is applied per group
            effective_wd = 0 # WD is handled by group setting
            if lr_decay_params and args.weight_decay > 0:
                 print("Note: Weight decay applied per-group in FusedAdam.")

            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=adam_w_mode, weight_decay=effective_wd, amsgrad=False)


    # --- deepspeed_offload property (keep as is) ---
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        is_deepspeed = isinstance(strategy, DeepSpeedStrategy)
        if is_deepspeed:
            # Access config safely
            try:
                cfg = strategy.config["zero_optimization"]
                return cfg.get("offload_optimizer") or cfg.get("offload_param")
            except KeyError:
                 # Handle cases where zero_optimization might not be configured
                 # Or if strategy.config is structured differently
                 print("Warning: Could not determine DeepSpeed offload status from strategy config.")
                 return False # Default to False if config is missing/unexpected
        return False

    # --- forward method (keep as is, but ensure v_first logic is sound) ---
    def forward(self, idx, recursive_depth=1): # Default depth to 1 if not generation
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, f"Input sequence length ({T}) exceeds model context length ({args.ctx_len})."

        x = self.emb(idx)

        # Initialize v_first state ONCE before the layers/recursion
        # The shape should match the expected input to the first Tmix/Block
        v_first_state = torch.zeros_like(x) # Initialize with zeros, common practice

        # --- Start: Handle Recursion ---
        # The original forward seems to mix layer iteration and recursion.
        # Let's clarify: We iterate through layers. Recursion (if > 1) means feeding the output back as input.

        current_input = x
        for step in range(recursive_depth):
            # Reset or carry over v_first state *per recursive step* if needed.
            # For standard training/inference, v_first is usually managed *within* the layer loop.
            # If recursion means full re-computation, v_first should be re-initialized or managed carefully.
            # Let's assume v_first is state *within* the layer pass for a given input.
            step_v_first = v_first_state # Use the same initial state for each block pass

            for block_idx, block in enumerate(self.blocks):
                if args.grad_cp == 1:
                    current_input, step_v_first = torch.utils.checkpoint.checkpoint(
                        block, current_input, step_v_first, use_reentrant=False
                    )
                else:
                    current_input, step_v_first = block(current_input, step_v_first)

            # After processing all blocks, current_input is the output of this recursive step.
            # For the next step (if any), this becomes the input.
            # v_first_state management across steps depends on the intended recursion logic.
            # If recursion simulates longer sequences, state *should* persist.
            # If it's just repeated processing, maybe reset. Assuming state persistence:
            # v_first_state = step_v_first # Update the state for the next potential step

        # --- End: Handle Recursion ---

        # Final output processing after all layers (and potential recursion)
        x = self.ln_out(current_input) # Use the final output from layers/recursion
        x = self.head(x)
        return x

    # --- MODIFIED training_step ---
    def training_step(self, batch, batch_idx):
        idx, targets = batch  # idx: (B, T), targets: (B, T)
        B, T = idx.shape

        # Define token IDs (using instance variables set in __init__)
        output_start_id = self.output_start_token_id
        output_end_id = self.output_end_token_id
        task_end_id = self.task_end_token_id
        ignore_idx = self.ignore_index

        # Create a mask for the targets tensor
        targets_masked = targets.clone()

        # Iterate over each sequence in the batch to apply masking
        for i in range(B):
            # Find the tokens in the original sequence (idx)
            current_idx_list = idx[i].tolist()

            # 1. Find the end of the last task
            try:
                # Search from right to left for </task>
                last_task_end_pos = len(current_idx_list) - 1 - current_idx_list[::-1].index(task_end_id)
                search_range = current_idx_list[:last_task_end_pos] # Sequence before last </task>
            except ValueError:
                # If no </task>, mask the whole sequence (or handle as error)
                targets_masked[i, :] = ignore_idx
                continue # Skip to next sequence in batch

            # 2. Find the start of the last <output> within this range
            try:
                # Search from right to left for <output> within the relevant range
                last_output_start_pos = len(search_range) - 1 - search_range[::-1].index(output_start_id)
            except ValueError:
                # If no <output> before last </task>, mask the whole sequence
                targets_masked[i, :] = ignore_idx
                continue

            # 3. Find the end of that specific <output> section
            try:
                # Search *after* the last_output_start_pos for the first </output>
                # The search starts from index last_output_start_pos + 1 in the original list
                target_output_end_pos = current_idx_list.index(output_end_id, last_output_start_pos + 1)

                # Ensure the found </output> is before the last </task>
                if target_output_end_pos >= last_task_end_pos:
                     raise ValueError("Found </output> is after or at </task>")

            except ValueError:
                 # If no </output> found after the start and before task end, mask the sequence
                 targets_masked[i, :] = ignore_idx
                 continue

            # 4. Create the mask: Ignore everything by default
            # We modify targets_masked directly. Initialize mask flags conceptually.
            current_mask = torch.ones_like(targets[i], dtype=torch.bool) # True means ignore

            # Indices in `idx` that are part of the target output:
            # range(last_output_start_pos + 1, target_output_end_pos)
            # Corresponding indices in `targets` to *keep* (not ignore):
            # range(last_output_start_pos, target_output_end_pos - 1)
            keep_start_in_targets = last_output_start_pos
            keep_end_in_targets = target_output_end_pos - 1 # The prediction *for* the token at target_output_end_pos is not needed

            # Apply the mask: Set positions *to keep* to False
            if keep_start_in_targets <= keep_end_in_targets: # Check for valid range
                current_mask[keep_start_in_targets : keep_end_in_targets + 1] = False # +1 because slice end is exclusive

            # Apply the conceptual mask to the actual targets tensor
            targets_masked[i, current_mask] = ignore_idx


        # --- Original forward pass and loss calculation ---
        train_depth = getattr(self.args, 'train_depth', 1) # Use configured depth
        logits = self(idx, recursive_depth=train_depth)

        # Calculate loss using the MASKED targets
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets_masked.view(-1),
                               ignore_index=ignore_idx) # Crucial: ignore_index

        # Log the loss (optional, PL handles it)
        self.log('train_loss', loss)

        # Optional: Calculate and log the percentage of tokens used for loss
        num_total_targets = targets.numel()
        num_masked_targets = (targets_masked == ignore_idx).sum().item()
        percent_active = 100.0 * (num_total_targets - num_masked_targets) / num_total_targets
        self.log('active_loss_percent', percent_active, prog_bar=True)


        return L2Wrap.apply(loss, logits)

    # --- validation_step (Should also use masking if val data has same format) ---
    def validation_step(self, batch, batch_idx):
        args = self.args
        idx, targets = batch
        B, T = idx.shape

        # --- Apply the SAME masking logic as in training_step ---
        output_start_id = self.output_start_token_id
        output_end_id = self.output_end_token_id
        task_end_id = self.task_end_token_id
        ignore_idx = self.ignore_index
        targets_masked = targets.clone()

        for i in range(B):
            current_idx_list = idx[i].tolist()
            try:
                last_task_end_pos = len(current_idx_list) - 1 - current_idx_list[::-1].index(task_end_id)
                search_range = current_idx_list[:last_task_end_pos]
            except ValueError:
                targets_masked[i, :] = ignore_idx
                continue
            try:
                last_output_start_pos = len(search_range) - 1 - search_range[::-1].index(output_start_id)
            except ValueError:
                targets_masked[i, :] = ignore_idx
                continue
            try:
                target_output_end_pos = current_idx_list.index(output_end_id, last_output_start_pos + 1)
                if target_output_end_pos >= last_task_end_pos:
                     raise ValueError("Found </output> is after or at </task>")
            except ValueError:
                 targets_masked[i, :] = ignore_idx
                 continue

            current_mask = torch.ones_like(targets[i], dtype=torch.bool)
            keep_start_in_targets = last_output_start_pos
            keep_end_in_targets = target_output_end_pos - 1
            if keep_start_in_targets <= keep_end_in_targets:
                current_mask[keep_start_in_targets : keep_end_in_targets + 1] = False
            targets_masked[i, current_mask] = ignore_idx
        # --- End of masking logic ---

        # Evaluate potentially at different depths
        eval_depths = getattr(args, 'eval_depths', [1])
        losses = {}
        for depth in eval_depths:
            logits = self(idx, recursive_depth=depth)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets_masked.view(-1), # Use masked targets for val loss
                                   ignore_index=ignore_idx)

            loss_key = f'val_loss_depth_{depth}'
            ppl_key = f'val_ppl_depth_{depth}'
            losses[loss_key] = loss
            # Avoid calculating exp(loss) if loss is NaN or Inf due to masking issues
            if torch.isfinite(loss):
                 losses[ppl_key] = torch.exp(loss)
            else:
                 losses[ppl_key] = float('inf') # Or some indicator value


        # Log collected losses
        # Ensure keys are unique if multiple depths are used
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Return a representative loss (e.g., for the first depth evaluated)
        # Handle potential case where the key might not exist if eval_depths is empty
        rep_loss_key = f'val_loss_depth_{eval_depths[0]}' if eval_depths else 'val_loss' # Fallback key
        return losses.get(rep_loss_key, torch.tensor(float('nan'))) # Return NaN if key missing


    # --- validation_epoch_end (keep as is or add custom aggregation) ---
    def validation_epoch_end(self, outputs):
        # outputs is a list of dictionaries returned by validation_step
        # Example: Calculate average over batches for a specific depth
        if outputs: # Check if outputs list is not empty
            target_key = f'val_loss_depth_{getattr(self.args, "eval_depths", [1])[0]}' # Key for the first eval depth
            valid_losses = [x[target_key] for x in outputs if isinstance(x, dict) and target_key in x and torch.isfinite(x[target_key])]
            if valid_losses:
                 avg_loss = torch.stack(valid_losses).mean()
                 self.log('avg_val_loss', avg_loss, sync_dist=True)
                 self.log(f'avg_{target_key}', avg_loss, sync_dist=True) # Log specific avg too
                 print(f"\nValidation Epoch End: Average {target_key} = {avg_loss:.4f}")
            else:
                 print(f"\nValidation Epoch End: No valid losses found for {target_key}.")
        else:
            print("\nValidation Epoch End: No outputs received.")
        pass


    # --- training_step_end (keep as is) ---
    def training_step_end(self, batch_parts):
        # This seems intended for custom loss aggregation across GPUs if needed.
        # Check if PL handles this automatically or if specific logic is required here.
        # If using DDP, PL usually handles gradient synchronization.
        # `all_gather` might be for logging/collecting metrics from all ranks.
        try:
            # Ensure batch_parts is gatherable (e.g., a tensor or dict of tensors)
            if batch_parts is not None:
                all_gathered_parts = self.all_gather(batch_parts)
                if self.trainer.is_global_zero:
                    # Assign the gathered losses (or batch parts) to the trainer attribute
                    self.trainer.my_loss_all = all_gathered_parts
                    # You might add other processing here if needed later
            # else:
            #      print("Warning: training_step_end received None batch_parts.")
        except Exception as e:
            print(f"Error in training_step_end during all_gather: {e}")
            # Potentially log error or handle gracefully
        pass # Keep pass if no further action needed


    # --- generate_init_weight (keep as is) ---
    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        # Iterate through state_dict which contains all registered parameters and buffers
        for n, p in self.state_dict().items():
            # Get the actual parameter object to check requires_grad if needed
            # param_obj = self.get_parameter(n) if '.' not in n else None # Simple check
            # A more robust way might be needed for nested modules

            shape = p.shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""

            # Check if parameter requires gradients (useful for debugging init)
            # requires_grad_str = ""
            # try: # Use try-except for robustness
            #     param_obj = self.get_parameter(n.replace('.', '.blocks.')) # Heuristic path
            #     if param_obj is not None:
            #          requires_grad_str = f"(req_grad={param_obj.requires_grad})"
            # except AttributeError: pass

            # print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n} {requires_grad_str}", end="")
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="") # Simpler print


            # --- Initialization logic based on parameter name ---
            scale = 1.0
            init_method = "default" # Keep track of how it was initialized

            # Conditions for skipping custom init or using specific scales
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                # Specific handling for LayerNorm weights
                if 'ln_x.weight' in n and hasattr(self, 'args'): # Check if args exists
                     try:
                         layer_id = int(n.split('.')[1]) # Assumes format blocks.N.ln_x.weight
                         layer_scale = (1 + layer_id) / self.args.n_layer
                         m[n] = (p.data.clone() * 0.0) + (layer_scale ** 0.7)
                         init_method = "ln_x custom"
                     except (IndexError, ValueError, AttributeError):
                         print(f" [WARN: Could not parse layer ID for {n}, using default init]", end="")
                         m[n] = p.data.clone() # Fallback to default
                         init_method = "default (error in ln_x)"
                else:
                    m[n] = p.data.clone() # Keep original value for these
                    init_method = "keep original"
                print(f" [{init_method}]")

            elif n == "emb.weight":
                m[n] = p.data.clone()
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                init_method = f"uniform {scale}"
                print(f" [{init_method}]")

            elif n == "head.weight":
                m[n] = p.data.clone()
                scale = 0.5
                if hasattr(self.args, 'vocab_size') and hasattr(self.args, 'n_embd'):
                    if self.args.vocab_size > self.args.n_embd:
                         scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                     print(" [WARN: vocab_size or n_embd not in args, using default head scale 0.5]", end="")
                nn.init.orthogonal_(m[n], gain=scale)
                init_method = f"orthogonal {scale:.2f}"
                print(f" [{init_method}]")

            else:
                # Default for other weights (mostly linear layers)
                if not n.endswith('.weight'):
                     print(" [WARN: Unexpected parameter format, skipping init]", end="")
                     m[n] = p.data.clone()
                     init_method = "skip (unexpected format)"
                     print(f" [{init_method}]")
                     continue # Skip to next parameter

                # Determine scale based on name patterns
                scale = 1.0 # Default orthogonal scale
                zero_out = False

                # Patterns for zero initialization
                zero_patterns = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                if any(kk in n for kk in zero_patterns):
                    scale = 0
                    zero_out = True
                    init_method = "zero"

                # Patterns for smaller scale initialization
                small_scale_patterns = {".att.key.": 0.1, ".att.gate.": 0.1} # Example scales
                if not zero_out:
                     for kk, specific_scale in small_scale_patterns.items():
                         if kk in n:
                             scale = specific_scale
                             init_method = f"orthogonal {scale}"
                             break # Use the first matching pattern
                     else: # If no specific pattern matched
                          init_method = f"orthogonal {scale}"

                print(f" [{init_method}]")

                # Perform initialization
                # Allocate on CPU first, then maybe move based on config
                m[n] = torch.empty_like(p.data, device='cpu') # Init on CPU

                if zero_out:
                    nn.init.zeros_(m[n])
                elif scale < 0: # Should correspond to uniform init logic above
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else: # Orthogonal init
                    nn.init.orthogonal_(m[n], gain=scale)


            # --- End Initialization Logic ---

            # Ensure initialized tensor is on CPU and has correct dtype
            m[n] = m[n].cpu()
            float_mode = os.environ.get("RWKV_FLOAT_MODE", "fp32") # Default to fp32
            if float_mode == "fp16":
                m[n] = m[n].half()
            elif float_mode == "bf16":
                m[n] = m[n].bfloat16()
            # Else keep float32

            n_params += m[n].numel()

        print(f'\nTotal model parameters initialized: {n_params / 1e6:.2f} M')
        gc.collect()
        # Avoid calling torch.cuda.empty_cache() here, let PyTorch manage memory.
        return m