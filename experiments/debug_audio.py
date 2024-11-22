import requests
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, WhisperProcessor, AutoConfig
from transformers.pipelines.audio_utils import ffmpeg_read


MODEL_PATH = "/workspace/exp-kunato/vllm/pretrained/DiVA-llama-3.2-1b"


def audio_batch():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = WhisperProcessor.from_pretrained(config.reference_encoder)
    tokenizer = AutoTokenizer.from_pretrained(config.reference_decoder)

    conversation1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                },
                {"type": "text", "text": "<|reserved_special_token_0|>"},
            ],
        },
    ]

    conversation2 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac",
                },
                {"type": "text", "text": "<|reserved_special_token_0|>"},
            ],
        },
    ]

    conversations = [conversation1, conversation2]

    audios = []
    texts = []
    for conversation in conversations:
        audio_infos_vllm = []
        text_infos_vllm = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_infos_vllm.append(
                            (
                                ffmpeg_read(
                                    requests.get(ele["audio_url"]).content,
                                    sampling_rate=processor.feature_extractor.sampling_rate,
                                ),
                                processor.feature_extractor.sampling_rate,
                            )
                        )
                    elif ele["type"] == "text":
                        text_infos_vllm.append(
                            {"role": message["role"], "content": ele["text"]}
                        )
            else:
                assert isinstance(message["content"], str)
                text_infos_vllm.append(
                    {"role": message["role"], "content": message["content"]}
                )
        texts.append(
            tokenizer.apply_chat_template(
                text_infos_vllm,
                add_generation_prompt=True,
                tokenize=False,
            )
        )
        audios.append(audio_infos_vllm)

    inputs = [
        {
            "prompt": texts[i],
            "multi_modal_data": {"audio": audios[i]},
        }  # audio is array of byte
        for i in range(len(conversations))
    ]
    return inputs


def main():
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.5,
        max_num_seqs=8, # required as it mistaken using bs = 500 and oom
        enforce_eager=True, # Disable CUDA graph, force call forward in every decode step.
        limit_mm_per_prompt={"audio": 1},
        max_model_len=4096,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.01,
        top_k=1,
        repetition_penalty=1.1,
        max_tokens=256,
    )

    inputs = audio_batch()
    print(f"{inputs=}")

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print("=" * 40)
        print(f"Inputs[{i}]: {inputs[i]['prompt']}")
        print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()
