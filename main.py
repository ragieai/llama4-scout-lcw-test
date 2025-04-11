from transformers import Llama4ForConditionalGeneration, AutoTokenizer
import torch
import time

file = "transcripts_4MB.txt"
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

PROMPT = """
Summarize the following text below.  Your response should be a concise summary of the text.

{text}
"""

def render_prompt(text):
    return PROMPT.format(text=text)

def main():
    with open(file, "r") as f:
        very_long_text = "\n".join(f.readlines())

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = Llama4ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flex_attention",
        torch_dtype=torch.bfloat16
    )

    messages = [
        {"role": "user", "content": render_prompt(very_long_text)}
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    torch.cuda.synchronize()
    start = time.time()
    out = model.generate(
        input_ids.to(model.device),
        prefill_chunk_size=2048*8,
        max_new_tokens=300,
        cache_implementation="hybrid",
    )
    print(time.time()-start)
    print(tokenizer.batch_decode(out[:, input_ids.shape[-1]:]))
    print(f"{torch.cuda.max_memory_allocated(model.device) / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
