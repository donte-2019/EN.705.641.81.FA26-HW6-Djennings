"""
Few-shot BoolQ evaluation for HW6 section 3.3.4.

The assignment uses gpt-3.5-turbo-instruct, which has been retired,
so I use gpt-4o-mini through Chat Completions. The script selects
4 yes and 4 no examples from the training set, alternates them in
the prompt, and evaluates 30 validation questions.

Set your API key before running:
    export OPENAI_API_KEY=<your key>
    python openai_gpt-boolq.py
"""
import random

from datasets import load_dataset
from openai import OpenAI

MODEL = "gpt-4o-mini"
N_DEMOS = 8      # demonstrations in the prompt (half yes, half no)
N_EVAL = 30      # validation questions to evaluate
SEED = 0

client = OpenAI()  # reads OPENAI_API_KEY from the environment
boolq = load_dataset("boolq")
rng = random.Random(SEED)


def format_example(example, with_answer):
    text = f"Passage: {example['passage']}\nQuestion: {example['question']}?\nAnswer:"
    if with_answer:
        text += " yes" if example["answer"] else " no"
    return text


# Pick 4 random yes and 4 random no examples, then alternate them in the prompt.
train = boolq["train"]
yes_ids = rng.sample([i for i, a in enumerate(train["answer"]) if a], N_DEMOS // 2)
no_ids = rng.sample([i for i, a in enumerate(train["answer"]) if not a], N_DEMOS // 2)
demo_ids = [i for pair in zip(yes_ids, no_ids) for i in pair]

context = "Answer each question about the passage with yes or no.\n\n"
context += "\n\n".join(format_example(train[i], with_answer=True) for i in demo_ids) + "\n\n"

# evaluate on a random sample of the validation set
validation = boolq["validation"]
eval_ids = rng.sample(range(len(validation)), N_EVAL)

correct = 0
unparsed = 0
for n, i in enumerate(eval_ids, start=1):
    example = validation[i]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": context + format_example(example, with_answer=False)}],
        temperature=0,
        max_tokens=3,
    )
    reply = response.choices[0].message.content.strip().lower()
    if reply.startswith("yes"):
        prediction = True
    elif reply.startswith("no"):
        prediction = False
    else:
        prediction = None
        unparsed += 1
    correct += prediction == example["answer"]
    print(f"{n:2d}. gold={str(example['answer']):5s} predicted={str(prediction):5s} ({reply!r})  {example['question']}")

gold_yes = sum(validation[i]["answer"] for i in eval_ids)
print(f"\nmodel: {MODEL}, demonstrations: {N_DEMOS} ({N_DEMOS // 2} yes / {N_DEMOS // 2} no), "
      f"evaluated: {N_EVAL} ({gold_yes} yes / {N_EVAL - gold_yes} no), unparsed replies: {unparsed}")
print(f"accuracy = {correct / N_EVAL:.4f} ({correct}/{N_EVAL})")
