# PowerShell script converted from demo-training-prepare.sh

# Set parameters
$MODEL_TYPE = "x070" # x060 => rwkv-6.0
$N_LAYER = "8"
$N_EMBD = "320"
$CTX_LEN = "512" # !!! change magic_prime if you change ctx_len !!!
$PROJ_DIR = "out/L$N_LAYER-D$N_EMBD-$MODEL_TYPE" # set output folder

# Create data folder if not exists
if (!(Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

# Download minipile files
#Invoke-WebRequest -Uri "https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx" -OutFile "data/minipile.idx"
#Invoke-WebRequest -Uri "https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin" -OutFile "data/minipile.bin"

# Run training command
python train.py --load_model "0" --wandb "Test" --proj_dir $PROJ_DIR `
 --data_file "./train" --data_type "binidx" --vocab_size 65536 --my_testing $MODEL_TYPE `
 --ctx_len $CTX_LEN --train_stage 1 --epoch_count 1 --epoch_begin 0 `
 --epoch_save 1 --weight_decay 0 --head_size 64 `
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --my_exit_tokens 1498226207 --magic_prime 2926181 `
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 `
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1