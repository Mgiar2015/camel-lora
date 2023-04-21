import json

with open("fisher_new.json") as f:
    data = json.load(f)

entries = []
for entry in data:
   s = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{entry['instruction']}\n\n### Input:\n{entry['input']}\n\n### Response:\n"
   entries.append(s)

from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
# print(longest)


tokenizer.pad_token_id = (
   0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

def tokenize(prompt, add_eos_token=True):
   # there's probably a way to do this with the tokenizer settings
   # but again, gotta move fast
   result = tokenizer(
      prompt,
      truncation=True,
      padding=False,
      return_tensors=None,
   )
   if (
      result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token
   ):
      result["input_ids"].append(tokenizer.eos_token_id)
      result["attention_mask"].append(1)

   result["labels"] = result["input_ids"].copy()

   return result

count = 0
from tqdm import tqdm
max_len = -1
longest = None
for i in tqdm(range(len(entries))):
   e = entries[i]
   if max_len < len(e):
      max_len = len(e)
      longest = e
   # tok_e = tokenize(e)
   # if len(tok_e['input_ids']) > 512:
   #    count += 1

toks = tokenize(longest)['input_ids']
h_toks = tokenize(longest[:len(longest)//2])['input_ids']
print(len(toks), len(longest))
# print(count/len(entries))
breakpoint()
# print(count/len(entries))