import sys
import os
"""
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
"""
import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter

load_8bit: bool = False
base_model: str = "decapoda-research/llama-7b-hf"
lora_weights: str = "camel-alpaca-fisher"
prompt_template: str = "alpaca"  # The prompt template to use, will default to alpaca
person_a= "Person A lives in New Jersey, loves to cook, and works as a yoga trainer for clients in new york city. They are 29 years old, a woman and get really excited by travel"
person_b=   "Person B lives in San Francisco, loves to hike, and works as a software engineer at Microsoft. He is 26 years old, a man and excited about artificial intelligence"
convo=True
temperature=0.8
top_p=0.6
top_k=40
num_beams=2
max_new_tokens=128

prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

p_a = input("Give Person A's personality, otherwise it will use default\n")
p_a = person_a if (p_a is None or len(p_a) < 3) else p_a
p_b = input("Give Person B's personality, otherwise it will use default\n")
p_b = person_b if (p_b is None or len(p_b) < 3) else p_b
conversation_history = input("Give the history (optional), e.g. A: Hey how are you doing? B: I'm great, how are you?")
prompt = f"Generate interesting and casual conversation between speaker A based on Speaker A's personality. Person A: {p_a}\n"
prompt += conversation_history
breakpoint()

while True:
    print("======= GENERATION ========")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=int(len(prompt)*0.7) + max_new_tokens,
        )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    if output.startswith(prompt):
        output = output[len(prompt):]

    import re

    pattern = r'^(A|B):.*$'
    match = re.search(pattern, output, re.MULTILINE)
    if match:
        utterance = match.group()
        prompt += f"\n{utterance}\n"
        breakpoint()
    else:
        print("doesn't start with the prompt")
        breakpoint()

    your_reply = input("Reply, make sure to add A/B. 'q' to quit")
    prompt+=f"{your_reply}\n"
    breakpoint()
    if input_str == 'q':
        break


# if __name__ == "__main__":
#     main()
