# diag.py  — usage: python diag.py <model_id> <gptq|compressed-tensors|none> <raw|chat> [eager]
import sys
from vllm import LLM, SamplingParams
from datasets import load_dataset

model, quant, mode = sys.argv[1:4]
eager = len(sys.argv) > 4 and sys.argv[4] == "eager"

prompt = load_dataset("openai/openai_humaneval")["test"][0]["prompt"]
llm = LLM(model=model, quantization=None if quant == "none" else quant,
          dtype="float16", max_model_len=2048, gpu_memory_utilization=0.8,
          enforce_eager=eager)
if mode == "chat":
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": f"Complete this function:\n\n{prompt}"}],
        add_generation_prompt=True, tokenize=False)

out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=200, logprobs=5))[0].outputs[0]
print(repr(out.text[:300]))
print(out.finish_reason, len(out.token_ids))
print(out.logprobs[0])