import argparse
import json
import os
import time

import openai
openai.api_key = "XXXX"
import tqdm

import shortuuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 5
REQ_TIME_GAP = 10


def get_eval(sys_prompt, user_prompt: str, max_tokens: int):
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"


def parse_score(review):
    try:
        coherence_score = review.split("\n")[0]
        consistent_score = review.split("\n")[1]
        fluency_score = review.split("\n")[2]
        relevance_score = review.split("\n")[3]
        score = {}
        score["coherence"] = [float(coherence_score.split(" ")[0]), float(coherence_score.split(" ")[1])]
        score["consistent"] = [float(consistent_score.split(" ")[0]), float(consistent_score.split(" ")[1])]
        score["fluency"] = [float(fluency_score.split(" ")[0]), float(fluency_score.split(" ")[1])]
        score["relevance"] = [float(relevance_score.split(" ")[0]), float(relevance_score.split(" ")[1])]
        return score
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
    return [-1, -1]

def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # Default to general category (index=0)
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    prompt_json = prompt_jsons[prompt_id - 1]
    assert prompt_json["prompt_id"] == prompt_id

    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    return sys_prompt, prompt, reviewer_idx + 1


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question-file", default="./table/summary_question.jsonl")
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    parser.add_argument("-p", "--prompt-file", default="./table/summary_prompt.jsonl")
    parser.add_argument("-r", "--reviewer-file", default="./table/reviewer.jsonl")
    parser.add_argument("-o", "--output-review-file", type=str, default="./commit_logs/summarization/judge_random_14.jsonl")
    parser.add_argument("-i1", "--input-file1", type=str, default="./commit_logs/summarization/cosine_pattern_0per_False.jsonl")
    parser.add_argument("-i2", "--input-file2", type=str, default="./commit_logs/judge_random_14.jsonl")
    parser.add_argument('--sparsity_ratio', type=float, default=0.0, help='Sparsity level')
    parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", default="magnitude" ,type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    print(args)

    # args.answer_file_list.append("./table/gpt35_answer.jsonl")
    # args.answer_file_list.append("./commit_logs/random_14.jsonl")
    # args.output_review_file = filename + "_evaluation.jsonl"

    question_jsons = get_json_list(args.question_file)
    answer1_jsons = get_json_list(args.input_file1)
    answer2_jsons = get_json_list(args.input_file2)
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    # check if # of questions, answers are the same
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    handles = []
    review_jsons = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))

    import tqdm
    for i in question_idx_list:
        assert (
            answer1_jsons[i]["question_id"]
            == question_jsons[i]["question_id"]
            == answer2_jsons[i]["question_id"]
        )

        ques = question_jsons[i]["text"]
        cat = question_jsons[i]["category"]
        ans1 = answer1_jsons[i]["text"]
        ans2 = answer2_jsons[i]["text"]
        sys_prompt, prompt, reviewer_id = gen_prompt(
            reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2
        )
        review_id = shortuuid.uuid()
        review_jsons.append(
            {
                "review_id": review_id,
                "question_id": question_jsons[i]["question_id"],
                "answer1_id": answer1_jsons[i]["answer_id"],
                "answer2_id": answer2_jsons[i]["answer_id"],
                "reviewer_id": reviewer_id,
                "metadata": {},
            }
        )
        # To avoid the rate limit set by OpenAI
        handles.append(get_eval(sys_prompt, prompt, args.max_tokens))
        logger.info(
            f"Waiting for {REQ_TIME_GAP} seconds before sending the next request."
        )
        time.sleep(REQ_TIME_GAP)

    reviews = handles
    chat_gpt_score = 0
    ours_score = 0
    count = 0
    with open(f"{args.output_review_file}", "w") as output_review_file:
        for idx, review in enumerate(reviews):
            scores = parse_score(review)
            review_jsons[idx]["text"] = review
            review_jsons[idx]["score"] = scores
            output_review_file.write(json.dumps(review_jsons[idx]) + "\n")
