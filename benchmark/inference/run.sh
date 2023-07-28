export CUDA_VISIBLE_DEVICES=0
python chatglm2-6b.py --model_dir /data/chatglm2-6b --bits 16 --debug
python chatglm2-6b.py --model_dir /data/chatglm2-6b --bits 8
python chatglm2-6b.py --model_dir /data/chatglm2-6b --bits 4