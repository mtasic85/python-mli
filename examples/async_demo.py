import asyncio

from mli import AsyncMLIClient


async def async_demo():
    async_client = AsyncMLIClient('http://127.0.0.1:5000', 'ws://127.0.0.1:5000')

    print(await async_client.text(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "sample_len": 512,
        "prompt": "Building a website can be done in 10 simple steps:\nStep 1:"
    }))

    print(await async_client.chat(**{
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

    async for chunk in async_client.iter_text(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "sample_len": 512,
        "prompt": "Building a website can be done in 10 simple steps:\nStep 1:"
    }):
        print(chunk, sep='', end='', flush=True)

    async for chunk in async_client.iter_chat(**{
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


if __name__ == '__main__':
    asyncio.run(async_demo())
