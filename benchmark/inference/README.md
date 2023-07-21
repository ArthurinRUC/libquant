# 大模型推理性能测试

探索模型量化、各类算法和 kernel 的改进，是否能提升推理速度，能提升多少？

## 一、测试环境与设置

- 目前所有模型均在 1 张 V100 16GB 上测试，模型默认存储精度为 16bit
- transformers==4.30.2
- bitsandbytes==0.40.2



## 二、实验步骤

- 不同的「模型+参数设置」为一组实验；
- 每组实验都会测试推理预填充（prefill）和解码（decode）的速度；
- 每组实验的 prefill 测试和 decode 测试均跑 500 个 iters，输入 batch size 恒定为 1；
- Base 基准组参数设置：
- prefill 的输入长度均为 50 tokens
- decode 的 kv cache 长度均为 50 tokens



## 三、实验结果

| 模型          | 参数设置（Base 为基准组） | prefill 速度（token/s） | decode 速度（token/s） | 输出每 token 的 latency（ms） | 显存占用  |
| ------------- | ------------------------- | ----------------------- | ---------------------- | ----------------------------- | --------- |
| ChatGLM2-6b   | Base                      | 1,270.85                | 27.09                  | 36.92                         | 12,887 MB |
|               | load_in_8bit              | 408.62                  | 8.97                   | 111.52                        | 8,127 MB  |
|               | load_in_4bit              | 337.49                  | 18.42                  | 54.28                         | 5,859 MB  |
| ChatGLM-6b    | Base                      | 724.44                  | 20.85                  | 47.96                         | 12,473 MB |
|               | load_in_8bit              | 348.67                  | 8.03                   | 124.58                        | 8,699 MB  |
| load_in_4bit  | 266.11                    | 14.62                   | 68.42                  | 6,197 MB                      |           |
| Chinese-LLaMA | Base                      | 897.10                  | 22.05                  | 45.35                         |           |
| Llama-2       | Base                      |                         |                        |                               |           |

ChatGLM2 采用了 torch 2.0 加速版的 Attention（例如 Flashattention、efficient-memory attention等），而其他模型的 attention 没有新的改进；