# %% [markdown]
# # Part 3 Starter Notebook
#
# This notebook loads an AIME 2024 dataset, runs a model on each problem, extracts an AIME-style final answer, and grades the outputs.

# %%
import re

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# %%
MODEL_NAME = "Qwen/Qwen3-4B"
# or allenai/Olmo-3-7B-Thinking
DATASET_NAME = "OpenRLHF/aime-2024"
MAX_NEW_TOKENS = 1024

# %% [markdown]
# ## Loading the model and the data

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = LLM(model=MODEL_NAME, gpu_memory_utilization=0.9, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(DATASET_NAME, split="train")

# %% [markdown]
# ## Evaluation helpers


# %%
def strip_thinking_trace(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", text, flags=re.DOTALL
    )
    return text.strip()


def extract_answer(text: str, mode="exact_match") -> int | None:
    """Extract an AIME-style integer answer from a model completion."""
    answer_text = strip_thinking_trace(text)
    if not answer_text:
        if mode == "exact_match":
            return None
        else:
            answer_text = text  # fall back to full text

    # 1. Boxed LaTeX answer: \boxed{123}
    if mode == "exact_match":
        boxed = re.findall(r"\\boxed\{(\d+)\}", answer_text)
        if boxed:
            val = int(boxed[-1])
            return val
        else:
            return None

    elif mode == "flexible_extract":
        # 2. "The answer is N" or "answer: N" patterns
        patterns = [
            r"(?:the\s+)?answer\s+is\s+[:\s]*(\d+)",
            r"answer[:\s]+(\d+)",
            r"=\s*(\d+)\s*$",
            r"(?:therefore|thus|so),?\s+(\d+)\s*(?:\.|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            if matches:
                val = int(matches[-1])
                return val

        # 3. Last integer in [0, 999] in the answer portion
        integers = re.findall(r"\b(\d{1,3})\b", answer_text)
        for candidate in reversed(integers):
            val = int(candidate)
            return val
        return None


# %% [markdown]
# ## Inference
#
# You can also explore using vLLM to speed up inference!

# %%
ENABLE_THINKING = True  # Set False for no-thinking condition
ANSWER_MODE = "exact_match"


# %%
def build_prompt(problem, enable_thinking):
    messages = [
        {
            "role": "system",
            "content": "You are a careful competition math assistant.  Always output your final answer in \\boxed{}.",
        },
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def generate_greedy(model, prompt, max_tokens=MAX_NEW_TOKENS):
    # Greedy decoding
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    outputs = model.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


def generate_fixed_budget(model, tokenizer, prompt, budget):
    stop_token_ids = tokenizer.convert_tokens_to_ids(["</think>", "<|im_end|>"])

    current_thinking_tokens = 0
    while current_thinking_tokens < budget:
        remaining_budget = budget - current_thinking_tokens

        # Recommended sampling parameters on model card at https://huggingface.co/Qwen/Qwen3-4B
        sampling_params = SamplingParams(
            max_tokens=remaining_budget,
            stop_token_ids=stop_token_ids,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
        )

        outputs = model.generate([prompt], sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        num_generated_tokens = len(outputs[0].outputs[0].token_ids)

        prompt += generated_text
        current_thinking_tokens += num_generated_tokens

        if current_thinking_tokens < budget:
            prompt += "\nWait"
            # print("INJECTING WAIT TOKEN!!!")
            current_thinking_tokens += len(
                tokenizer.encode("\nWait", add_special_tokens=False)
            )

    prompt += "\n</think>"
    # print("INJECTING STOP TOKEN!!!")

    # Recommended sampling parameters on model card at https://huggingface.co/Qwen/Qwen3-4B
    final_answer_params = SamplingParams(
        max_tokens=4000, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
    )

    final_answer_outputs = model.generate([prompt], final_answer_params, use_tqdm=False)
    return prompt + final_answer_outputs[0].outputs[0].text


def generate_fixed_budget_batched(model, tokenizer, prompts, budget):
    prompts = [prompt + "<think>" for prompt in prompts]
    remaining_budget = budget - len(
        tokenizer.encode("<think>", add_special_tokens=False)
    )

    stop_token_ids = tokenizer.convert_tokens_to_ids(
        ["<think>", "</think>", "<|im_end|>"]
    )

    # Recommended sampling parameters on model card at https://huggingface.co/Qwen/Qwen3-4B
    sampling_params = SamplingParams(
        max_tokens=remaining_budget,
        stop_token_ids=stop_token_ids,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        min_tokens=remaining_budget,
    )

    outputs = model.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    prompts = [
        prompt + generated_texts[i] + "\n</think>" for i, prompt in enumerate(prompts)
    ]

    # Recommended sampling parameters on model card at https://huggingface.co/Qwen/Qwen3-4B
    final_answer_params = SamplingParams(
        max_tokens=4000, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
    )

    final_answer_outputs = model.generate(prompts, final_answer_params)
    generated_texts = [output.outputs[0].text for output in final_answer_outputs]
    return [prompt + generated_texts[i] for i, prompt in enumerate(prompts)]


def extract_thinking(text):
    match = re.search(r"<think>(.*)", text, re.DOTALL)
    if match:
        content = match.group(1)
        if "</think>" in content:
            content = content.split("</think>")[0]
        return content
    return ""


def count_thinking_tokens(text):
    t = extract_thinking(text)
    if not t:
        return 0
    return len(tokenizer.encode(t, add_special_tokens=False))


def count_tokens(text):
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def get_records(dataset, model_outputs, verbose=True):
    records = []

    for i, example in enumerate(dataset):
        problem = example["prompt"][0]["content"]
        gold_answer = int(example["label"])

        model_output = model_outputs[i]

        exact_extracted = extract_answer(model_output, mode="exact_match")
        flexible_extracted = extract_answer(model_output, mode="flexible_extract")
        exact_correct = (
            True
            if exact_extracted is not None and exact_extracted == gold_answer
            else False
        )
        flexible_correct = (
            True
            if flexible_extracted is not None and flexible_extracted == gold_answer
            else False
        )

        thinking_tokens = count_thinking_tokens(model_output)

        records.append(
            {
                "problem": problem,
                "gold_answer": gold_answer,
                "model_output": model_output,
                "exact_extracted": exact_extracted,
                "flexible_extracted": flexible_extracted,
                "exact_correct": exact_correct,
                "flexible_correct": flexible_correct,
                "thinking_tokens": thinking_tokens,
            }
        )

        if verbose:
            print(
                f"[{i + 1}/{len(dataset)}] gold={gold_answer} exact_pred={exact_extracted} flexible_pred={flexible_extracted} exact_correct={exact_correct} flexible_correct={flexible_correct}"
            )

    return records


# %%
# Build prompts for thinking and no thinking

prompts_think = [
    build_prompt(example["prompt"][0]["content"], True) for example in dataset
]
prompts_no_think = [
    build_prompt(example["prompt"][0]["content"], False) for example in dataset
]

# %%
# This cell is for 2.1 warm-up. However, the next cell below has an optimized version.

records = []

for i, example in enumerate(dataset):
    problem = example["prompt"][0]["content"]
    gold_answer = int(example["label"])

    prompt = build_prompt(problem, ENABLE_THINKING)

    model_output = generate_greedy(model, prompt)

    exact_extracted = extract_answer(model_output, mode="exact_match")
    flexible_extracted = extract_answer(model_output, mode="flexible_extract")
    exact_correct = (
        True
        if exact_extracted is not None and exact_extracted == gold_answer
        else False
    )
    flexible_correct = (
        True
        if flexible_extracted is not None and flexible_extracted == gold_answer
        else False
    )

    thinking_tokens = count_thinking_tokens(model_output)

    records.append(
        {
            "problem": problem,
            "gold_answer": gold_answer,
            "model_output": model_output,
            "exact_extracted": exact_extracted,
            "flexible_extracted": flexible_extracted,
            "exact_correct": exact_correct,
            "flexible_correct": flexible_correct,
            "thinking_tokens": thinking_tokens,
        }
    )

    print(
        f"[{i + 1}/{len(dataset)}] gold={gold_answer} exact_pred={exact_extracted} flexible_pred={flexible_extracted} exact_correct={exact_correct} flexible_correct={flexible_correct}"
    )

results_df = pd.DataFrame(records)
results_df

# %%
# This cell is the optimized version for 2.1 warm-up which uses batched prompts.

prompts = prompts_think if ENABLE_THINKING else prompts_no_think

# Greedy decoding
sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

outputs = model.generate(prompts, sampling_params)
model_outputs = [output.outputs[0].text for output in outputs]

records = get_records(dataset, model_outputs)

results_df = pd.DataFrame(records)
results_df

# %%
# Save results for 2.1 warm-up

results_df.to_csv(
    f"results_df_greedy_{MAX_NEW_TOKENS}_{'think' if ENABLE_THINKING else 'no_think'}.csv",
    index=False,
)
print(f"exact_correct accuracy: {results_df['exact_correct'].mean()}")
print(f"flexible_correct accuracy: {results_df['flexible_correct'].mean()}")

# %%
# Generate histogram of thinking lengths

print(results_df["thinking_tokens"].describe())
results_df["thinking_tokens"].hist()
plt.title("Histogram of Thinking Lengths")
plt.xlabel("Thinking Length (tokens)")
plt.show()

# %%
# This cell and the next 2 cells below can be used to incrementally generate and save results. It is not fully updated/maintained.

results_df_csv_file = "results_df_32000_v3.csv"  # Choose file to load
increment = 3  # Choose how many problems to do at a time
results_df = pd.read_csv(results_df_csv_file)
start_idx = len(results_df)
end_idx = min(start_idx + increment, len(dataset))
print(f"Resuming with start_idx: {start_idx} and end_idx: {end_idx}")

# %%
new_records = []

for i in range(start_idx, end_idx):
    problem = dataset[i]["prompt"][0]["content"]
    gold_answer = int(dataset[i]["label"])

    prompt = build_prompt(problem, ENABLE_THINKING)

    model_output = generate_greedy(model, prompt)

    extracted = extract_answer(model_output, mode=ANSWER_MODE)
    if extracted is not None:
        correct = extracted == gold_answer
    else:
        correct = False

    thinking_tokens = count_thinking_tokens(model_output)

    new_records.append(
        {
            "problem": problem,
            "gold_answer": gold_answer,
            "model_output": model_output,
            "extracted_answer": extracted,
            "correct": correct,
            "thinking_tokens": thinking_tokens,
        }
    )

    print(
        f"[{i + 1}/{len(dataset)}] gold={gold_answer} pred={extracted} correct={correct}"
    )

new_df = pd.DataFrame(new_records)
results_df = pd.concat([results_df, new_df], ignore_index=True)
results_df

# %%
results_df.to_csv(results_df_csv_file, index=False)
results_df["correct"].mean()

# %%
# This cell is for 2.2 scaling experiments (sequential). However, the next cell below has an optimized version.

thinking_budgets = [8000]

for budget in thinking_budgets:
    records = []

    for i, example in enumerate(dataset):
        problem = example["prompt"][0]["content"]
        gold_answer = int(example["label"])

        prompt = prompts_think[i]

        model_output = generate_fixed_budget(model, tokenizer, prompt, budget)

        exact_extracted = extract_answer(model_output, mode="exact_match")
        flexible_extracted = extract_answer(model_output, mode="flexible_extract")
        exact_correct = (
            True
            if exact_extracted is not None and exact_extracted == gold_answer
            else False
        )
        flexible_correct = (
            True
            if flexible_extracted is not None and flexible_extracted == gold_answer
            else False
        )

        thinking_tokens = count_thinking_tokens(model_output)

        records.append(
            {
                "problem": problem,
                "gold_answer": gold_answer,
                "model_output": model_output,
                "exact_extracted": exact_extracted,
                "flexible_extracted": flexible_extracted,
                "exact_correct": exact_correct,
                "flexible_correct": flexible_correct,
                "thinking_tokens": thinking_tokens,
            }
        )

        print(
            f"[{i + 1}/{len(dataset)}] gold={gold_answer} exact_pred={exact_extracted} flexible_pred={flexible_extracted} exact_correct={exact_correct} flexible_correct={flexible_correct}"
        )

    results_df = pd.DataFrame(records)
    results_df.to_csv(f"results_df_sequential_{budget}.csv", index=False)

# %%
# This cell is the optimized version for 2.2 sequential which uses batched prompts.

thinking_budgets = [32000]

for budget in thinking_budgets:
    model_outputs = generate_fixed_budget_batched(
        model, tokenizer, prompts_think, budget
    )

    records = get_records(dataset, model_outputs)

    results_df = pd.DataFrame(records)
    results_df.to_csv(f"results_df_sequential_{budget}.csv", index=False)

# %%
# Compute sequential results

sequential_thinking_tokens_list = [1024, 2000, 4000, 8000, 16000, 32000]

sequential_accs = {"exact": [], "flexible": []}
sequential_avg_tokens = []

for thinking_tokens in sequential_thinking_tokens_list:
    results_df = pd.read_csv(f"results_df_sequential_{thinking_tokens}.csv")
    sequential_accs["exact"].append(results_df["exact_correct"].mean())
    sequential_accs["flexible"].append(results_df["flexible_correct"].mean())
    sequential_avg_tokens.append(results_df["model_output"].apply(count_tokens).mean())

# %%
parallel_budget = 4000
m = 8

# %%
# This cell is the optimized version for 2.2 parallel which uses batched prompts.

for i in range(m):
    model_outputs = generate_fixed_budget_batched(
        model, tokenizer, prompts_think, parallel_budget
    )

    records = get_records(dataset, model_outputs, verbose=False)

    results_df = pd.DataFrame(records)
    results_df.to_csv(f"results_df_parallel_{i + 1}.csv", index=False)

# %%
# Combine the parallel results

results_dfs = []

for i in range(m):
    results_dfs.append(pd.read_csv(f"results_df_parallel_{i + 1}.csv"))

combined_results_df = pd.DataFrame(
    {
        "problem": results_dfs[0]["problem"],
        "gold_answer": results_dfs[0]["gold_answer"],
        "model_output_list": list(
            zip(*[results_df["model_output"] for results_df in results_dfs])
        ),
        "exact_extracted_list": list(
            zip(*[results_df["exact_extracted"] for results_df in results_dfs])
        ),
        "flexible_extracted_list": list(
            zip(*[results_df["flexible_extracted"] for results_df in results_dfs])
        ),
        "exact_correct_list": list(
            zip(*[results_df["exact_correct"] for results_df in results_dfs])
        ),
        "flexible_correct_list": list(
            zip(*[results_df["flexible_correct"] for results_df in results_dfs])
        ),
        "thinking_tokens_list": list(
            zip(*[results_df["thinking_tokens"] for results_df in results_dfs])
        ),
    }
)

combined_results_df

# %%
combined_results_df.to_csv("results_df_parallel_combined.csv", index=False)


# %%
def majority_vote(extracted_list):
    counts = Counter(extracted_list)
    return counts.most_common(1)[0][0]


def best_of_m(correct_list):
    return any(correct_list)


# %%
# Compute parallel results

combined_results_df = pd.read_csv("results_df_parallel_combined.csv")
m_vals = [1, 2, 4, 8]
parallel_thinking_tokens_list = [m_val * parallel_budget for m_val in m_vals]

parallel_accs = {
    "exact_majority": [],
    "flexible_majority": [],
    "exact_best_of_m": [],
    "flexible_best_of_m": [],
}
parallel_avg_tokens = []

for m_val in m_vals:
    print("Starting m_val: ", m_val)
    exact_majority_correct = []
    flexible_majority_correct = []
    exact_best_correct = []
    flexible_best_correct = []

    for _, row in combined_results_df.iterrows():
        gold_answer = row["gold_answer"]
        print("gold_answer: ", gold_answer)

        # Get first m completions
        print(type(row["exact_extracted_list"][:m_val]))
        exact_extracted_list = row["exact_extracted_list"][:m_val]
        flexible_extracted_list = row["flexible_extracted_list"][:m_val]
        exact_correct_list = row["exact_correct_list"][:m_val]
        flexible_correct_list = row["flexible_correct_list"][:m_val]

        # Majority vote
        exact_mv = majority_vote(exact_extracted_list)
        flexible_mv = majority_vote(flexible_extracted_list)
        exact_majority_correct.append(
            True if exact_mv is not None and exact_mv == gold_answer else False
        )
        flexible_majority_correct.append(
            True if flexible_mv is not None and flexible_mv == gold_answer else False
        )
        print("exact_mv: ", exact_mv)
        print("flexible_mv: ", flexible_mv)

        # Best of m
        exact_best_correct.append(best_of_m(exact_correct_list))
        flexible_best_correct.append(best_of_m(flexible_correct_list))

    parallel_accs["exact_majority"].append(np.mean(exact_majority_correct))
    parallel_accs["flexible_majority"].append(np.mean(flexible_majority_correct))
    parallel_accs["exact_best_of_m"].append(np.mean(exact_best_correct))
    parallel_accs["flexible_best_of_m"].append(np.mean(flexible_best_correct))
    # parallel_avg_tokens.append(np.mean([count_tokens(model_output) for model_output in combined_results_df["model_output_list"][:m_val]]))

# %%
# Exact scaling plot

plt.figure()

plt.plot(
    sequential_thinking_tokens_list,
    sequential_accs["exact"],
    marker="o",
    label="Sequential",
)
plt.plot(
    parallel_thinking_tokens_list,
    parallel_accs["exact_majority"],
    marker="o",
    label="Parallel: Majority Voting",
)
plt.plot(
    parallel_thinking_tokens_list,
    parallel_accs["exact_best_of_m"],
    marker="o",
    label="Parallel: Best-of-m",
)

plt.xlabel("Total Thinking Tokens Generated")
plt.ylabel("Accuracy")
plt.title("AIME-2024 Exact Accuracy with Various Scaling Strategies")
plt.legend()
plt.grid()

plt.show()

# %%
# Flexible scaling plot

plt.figure()

plt.plot(
    sequential_thinking_tokens_list,
    sequential_accs["flexible"],
    marker="o",
    label="Sequential",
)
plt.plot(
    parallel_thinking_tokens_list,
    parallel_accs["flexible_majority"],
    marker="o",
    label="Parallel: Majority Voting",
)
plt.plot(
    parallel_thinking_tokens_list,
    parallel_accs["flexible_best_of_m"],
    marker="o",
    label="Parallel: Best-of-m",
)

plt.xlabel("Total Thinking Tokens Generated")
plt.ylabel("Accuracy")
plt.title("AIME-2024 Flexible Accuracy with Various Scaling Strategies")
plt.legend()
plt.grid()

plt.show()

# %%
# Average tokens flexible plot

plt.figure()

plt.plot(
    sequential_avg_tokens, sequential_accs["flexible"], marker="o", label="Sequential"
)
plt.plot(
    parallel_avg_tokens,
    parallel_accs["flexible_majority"],
    marker="o",
    label="Parallel: Majority Voting",
)
plt.plot(
    parallel_avg_tokens,
    parallel_accs["flexible_best_of_m"],
    marker="o",
    label="Parallel: Best-of-m",
)

plt.xlabel("Average Total Tokens Generated")
plt.ylabel("Accuracy")
plt.title("AIME-2024 Flexible Accuracy vs Average Total Tokens Generated")
plt.legend()
plt.grid()

plt.show()

# %%
# Cells below this are old versions that are no longer being used

# %%
# New cell


def generate_with_budget_forcing(example, budget, model, tokenizer):
    # 1. Build the initial prompt using your existing template logic
    # This adds the system message, user problem, and the opening <think> tag
    prompt = build_prompt(example)

    # Identify the stop token for "end of thinking"
    # For Qwen/s1, this is usually </think> or a specific end-of-thought ID
    # stop_token_ids = tokenizer.convert_tokens_to_ids(["</think>", "<|im_end|>"])
    stop_token_ids = tokenizer.convert_tokens_to_ids(["</think>"])

    current_thinking_tokens = 0

    # 2. Sequential Scaling Loop
    while current_thinking_tokens < budget:
        remaining_budget = budget - current_thinking_tokens

        # We allow the model to generate up to the remaining budget
        # It will stop if it tries to output </think> or hits max_tokens
        sampling_params = SamplingParams(
            max_tokens=remaining_budget,
            stop_token_ids=stop_token_ids,
            temperature=0.0,
            skip_special_tokens=False,  # Keep tags so we can see them
        )

        output = model.generate([prompt], sampling_params, use_tqdm=False)[0]
        gen_text = output.outputs[0].text
        gen_tokens = len(output.outputs[0].token_ids)

        # Append the new reasoning to our prompt
        prompt += gen_text
        current_thinking_tokens += gen_tokens

        # LOGIC: Did it stop because it finished thinking or because it hit the budget?
        # If it finished early (stop reason was a stop token), inject "Wait"
        if current_thinking_tokens < budget:
            # Check if it actually emitted a stop tag; if so, we "Wait"
            # If it didn't emit a tag but stopped (rare), we still "Wait" to force length
            # prompt += " Wait,"
            prompt += "\nWait"
            # print("INJECTING WAIT TOKEN!!!")
            # current_thinking_tokens += len(tokenizer.encode(" Wait,", add_special_tokens=False))
            current_thinking_tokens += len(
                tokenizer.encode("\nWait", add_special_tokens=False)
            )

    # if prompt.endswith("</think>"):
    #     prompt = prompt[:-len("</think>")]

    if "</think>" in prompt:
        prompt = prompt.split("</think>", 1)[0]
        # print("REMOVING </think>")

    prompt += "\n</think>\n**Final Answer**\n"
    # print("INJECTING STOP TOKEN!!!")

    # Now generate the actual answer
    # We use a standard max_tokens here as the answer itself isn't part of the budget
    final_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        stop_token_ids=tokenizer.convert_tokens_to_ids(["<|im_end|>"]),
    )

    final_output = model.generate([prompt], final_params, use_tqdm=False)[0]
    return prompt + final_output.outputs[0].text


# %%
# New cell

thinking_budgets = [8000]

for budget in thinking_budgets:
    records_budget = []

    for i, example in enumerate(dataset):
        if i >= 5:
            break

        problem = example["prompt"][0]["content"]
        gold_answer = int(example["label"])

        # model_output = generate_with_budget(problem, budget, model, tokenizer)
        model_output = generate_with_budget_forcing(example, budget, model, tokenizer)

        extracted = extract_answer(model_output, mode=ANSWER_MODE)
        if extracted is not None:
            correct = extracted == gold_answer
        else:
            correct = False

        records_budget.append(
            {
                "problem": problem,
                "gold_answer": gold_answer,
                "model_output": model_output,
                "extracted_answer": extracted,
                "correct": correct,
            }
        )

        print(
            f"[{i + 1}/{len(dataset)}] gold={gold_answer} pred={extracted} correct={correct}"
        )

    results_budget_df = pd.DataFrame(records_budget)
    if ENABLE_THINKING:
        results_budget_df["thinking_tokens"] = results_budget_df["model_output"].apply(
            thinking_tokens
        )
        print(results_budget_df["thinking_tokens"].describe())
        results_budget_df["thinking_tokens"].hist()
        plt.title("Histogram of Thinking Lengths")
        plt.xlabel("Thinking Length (tokens)")
        plt.show()
    results_budget_df.to_csv(f"results_budget_{budget}_v2_df.csv", index=False)
    print(results_budget_df["correct"].mean())
