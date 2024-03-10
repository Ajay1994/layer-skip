import os
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
from typing import List, Optional

import fire
# torchrun --nproc_per_node 1 chat_generation.py  --max_seq_len 512 --max_batch_size 1
from llama import Llama, Dialog

def main(
    ckpt_dir = "/data/ajay_data/llama2_models/llama-2-7b-chat/",
    tokenizer_path = "/data/ajay_data/llama2_models/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        # [{"role": "user", "content": "Can you help me write a formal email to a potential business partner proposing a joint venture?"}]
        [{"role": "user", "content": "Can you plan a trip to Miami?"}]
    ]
    print(f"============> Number of layers: {generator.model.n_layers}")
    
    result = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"{dialogs[0][0]['role'].capitalize()}: {dialogs[0][0]['content']}\n")
    print(f"> {result[0]['generation']['role'].capitalize()}: {result[0]['generation']['content']}")

if __name__ == "__main__":
    fire.Fire(main)