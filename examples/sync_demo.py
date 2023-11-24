from mli import SyncMLIClient


def sync_demo_candle_stable_lm():
    sync_client = SyncMLIClient('http://127.0.0.1:5000')

    print(sync_client.text(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "sample_len": 512,
        "prompt": "Building a website can be done in 10 simple steps:\nStep 1:"
    }))

    print(sync_client.chat(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "sample_len": 512,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I need help building a website."},
            {"role": "assistant", "content": "Sure, let me know what and hwo do you need it built."},
            {"role": "user", "content": "Building a website can be done in 10 simple steps. Explain step by step."}
        ]
    }))

    for chunk in sync_client.iter_text(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "sample_len": 512,
        "prompt": "Building a website can be done in 10 simple steps:\nStep 1:"
    }):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "sample_len": 512,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I need help building a website."},
            {"role": "assistant", "content": "Sure, let me know what and hwo do you need it built."},
            {"role": "user", "content": "Building a website can be done in 10 simple steps. Explain step by step."}
        ]
    }):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_llama():
    sync_client = SyncMLIClient('http://127.0.0.1:5000')

    for chunk in sync_client.iter_text(**{
        "engine": "candle",
        "kind": "quantized",
        "model": "mistral-7b-v0.1.Q4_K_M.gguf",
        # "model": "yarn-llama-2-7b-128k.Q4_K_M.gguf",
        "sample_len": 8 * 1024,
        "prompt": "Building a perfect e-commerce website in 1234 simple steps:\nStep 1:"
    }):
        print(chunk, sep='', end='', flush=True)


if __name__ == '__main__':
    # sync_demo_candle_stable_lm()
    sync_demo_candle_llama()