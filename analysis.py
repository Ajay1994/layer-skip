import os
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
from typing import List, Optional

import fire
# torchrun --nproc_per_node 1 chat_generation.py  --max_seq_len 512 --max_batch_size 1
from llama import Llama, Dialog
'''
Transformer(
  (tok_embeddings): ParallelEmbedding()
  (layers): ModuleList(
    (0-39): 40 x TransformerBlock(
      (attention): Attention(
        (wq): ColumnParallelLinear()
        (wk): ColumnParallelLinear()
        (wv): ColumnParallelLinear()
        (wo): RowParallelLinear()
      )
      (feed_forward): FeedForward(
        (w1): ColumnParallelLinear()
        (w2): RowParallelLinear()
        (w3): ColumnParallelLinear()
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (output): ColumnParallelLinear()
)
'''
def layer_ffn_count(model):
    n_layers =  model.n_layers
    ffn = model.layers[0].feed_forward
    w1_count = ffn.w1.weight.shape[0] *  ffn.w1.weight.shape[1]
    w2_count = ffn.w2.weight.shape[0] *  ffn.w2.weight.shape[1]
    w3_count = ffn.w3.weight.shape[0] *  ffn.w3.weight.shape[1]

    return n_layers * (w1_count + w2_count + w3_count)

def layer_attention_count(model):
    n_layers =  model.n_layers
    attention = model.layers[0].attention
    wq_count = attention.wq.weight.shape[0] *  attention.wq.weight.shape[1]
    wk_count = attention.wk.weight.shape[0] *  attention.wk.weight.shape[1]
    wv_count = attention.wv.weight.shape[0] *  attention.wv.weight.shape[1]
    wo_count = attention.wo.weight.shape[0] *  attention.wo.weight.shape[1]

    return n_layers * (wq_count + wk_count + wv_count + wo_count)

def main(
    ckpt_dir = "/data/ajay_data/llama2_models/llama-2-7b-chat/",
    tokenizer_path = "/data/ajay_data/llama2_models/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("Model Created.")
    print(f"Number of layers: {generator.model.n_layers}")
    print(f"Total Number of FFN parameters: {layer_ffn_count(generator.model)}")
    print(f"Total Number of Attention parameters: {layer_attention_count(generator.model)}")


if __name__ == "__main__":
    fire.Fire(main)