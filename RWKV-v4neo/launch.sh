#/bin/bash
python train.py --load_model RWKV-4-Raven-3B-v9-EngOther-20230411-ctx4096.pth --proj_dir trains --data_file train.txt --data_type "utf-8" --vocab_size 50277 --ctx_len 4096 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 5 --micro_bsz 2 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 --lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.01 --lora_parts=att,time,ln