import os
from typing import List, Optional
import json

import fire
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25692 mt_bench_generation.py --random False --file
from llama import Llama, Dialog

def get_model_answers(generator, question, max_gen_len, temperature, top_p):
    ques_json = json.loads(question[0])
    dialogs: List[Dialog] = [
        # [{"role": "user", "content": "Can you help me write a formal email to a potential business partner proposing a joint venture?"}]
        [{"role": "user", "content": ques_json["text"]}]
    ]
    # print(f"============> Number of layers: {generator.model.n_layers}")
    
    result = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    # print(f"{dialogs[0][0]['role'].capitalize()}: {dialogs[0][0]['content']}\n")
    print(f"> {result[0]['generation']['role'].capitalize()}: {result[0]['generation']['content']}\n Skipped %: {generator.get_skip_ratio()}")
    import shortuuid
    ans_id = shortuuid.uuid()
    return {
            "question_id": ques_json["question_id"], 
            "text": result[0]['generation']['content'],
            "answer_id": ans_id,
            "model_id": "llama2-13b",
            "metadata": {"skip_ratio": generator.get_skip_ratio()},
            }


def run_eval(generator, question_file, answer_file, max_gen_len, temperature, top_p):
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    from tqdm import tqdm
    ans_handles = []
    for i in tqdm(range(0, len(ques_jsons))):
        ans_handles.append(
            get_model_answers(
                generator, ques_jsons[i : i + 1], max_gen_len, temperature, top_p
            )
        )

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_handles:
            ans_file.write(json.dumps(line) + "\n")

def main(
    ckpt_dir = "/data/ajay_data/llama2_models/llama-2-13b-chat/",
    tokenizer_path = "/data/ajay_data/llama2_models/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    random = True,
    file = "test"
):
    file_name = f"commit_logs/{file}_{random}.jsonl"

    print(file_name, random)
    from llama.utils import calc_skip_pattern
    skip_pattern = calc_skip_pattern(1, max_seq_len, random)
    
    args = {}
    args["file"] = file_name
    args["random"] = random
    args["skip_pattern"] = skip_pattern

    generator = Llama.build(
        args,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    run_eval(
        generator,
        "./table/question.jsonl",
        file_name,
        max_gen_len,
        temperature,
        top_p
    )

if __name__ == "__main__":
    fire.Fire(main)