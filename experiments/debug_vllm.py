import asyncio
import time
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm import LLM, AsyncLLMEngine
from transformers import AutoTokenizer, PreTrainedTokenizer

async def run_streaming(model_name: str, tokenizer: PreTrainedTokenizer):
    engine_args = AsyncEngineArgs(model=model_name, max_model_len=4096)
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    prompt = tokenizer.apply_chat_template("Please give me a travel iternary in Bangkok", tokenize=False, add_generation_prompt=True)
    results_generator = llm.generate(prompt, SamplingParams(top_p=0.95, temperature=1.0), str(time.monotonic()))
    outputs = ""
    async for request_output in results_generator:
        prompt = request_output.prompt
        if request_output.finished:
            print()
        else:
            out = request_output.outputs[-1].text
            print('hidden_state', request_output.outputs[-1].hidden_states.shape)
            if len(out) == 0:
                continue
            out_delta = out[len(outputs) :]
            print(out_delta)
            outputs = out
            
def run_sync(model_name: str, tokenizer: PreTrainedTokenizer):
    
    llm = LLM(model=model_name, max_model_len=4096)
    concurrent = 100
    texts = ["Please give me a travel iternary in Bangkok"] * concurrent
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        )
        for t in texts
    ]
    outputs = llm.generate(texts, sampling_params=SamplingParams(top_p=0.95, temperature=1.0))
    print("======" * 20)
    print("output ->", outputs[0])
    print("hidden state", outputs[0].outputs[0].hidden_states)
    

if __name__ == "__main__":
    # TODO support return multiple_token (unit_token / text token) -> text is the current one, unit is the optional and new one
    concurrent = 100
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # run_sync(model_name, tokenizer)
    asyncio.run(run_streaming(model_name, tokenizer))
