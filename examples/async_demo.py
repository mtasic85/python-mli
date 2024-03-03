import asyncio

from mli import AsyncMLIClient

from defs import (
    ENDPOINT,
    SYSTEM_TEXT,
    CAR_TEXT,
    CARS_TEXT,
    JSON_FLAT_ARRAY_GRAMMAR,
    JSON_FLAT_OBJECT_GRAMMAR,
)


async def async_demo_candle_stable_lm():
    async_client = AsyncMLIClient(ENDPOINT)

    async for chunk in async_client.iter_text(
        engine='candle',
        executable='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        sample_len=512,
        quantized=True,
        prompt='Building a website can be done in 10 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)

    async for chunk in async_client.iter_chat(
        engine='candle',
        executable='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        sample_len=512,
        quantized=True,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and hwo do you need it built.'},
            {'role': 'user', 'content': 'Building a website can be done in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    print(await async_client.text(
        engine='candle',
        executable='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        sample_len=512,
        prompt='Building a website can be done in 10 simple steps:\nStep 1:'
    ))

    print(await async_client.chat(
        engine='candle',
        executable='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        sample_len=512,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and hwo do you need it built.'},
            {'role': 'user', 'content': 'Building a website can be done in 10 simple steps. Explain step by step.'},
        ],
    ))


async def async_demo_llama_cpp_main_stable_lm():
    async_client = AsyncMLIClient(ENDPOINT)

    async for chunk in async_client.iter_text(
       engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/rocket-3B-GGUF',
        model='rocket-3b.Q4_K_M.gguf',
        sample_len=3 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    async for chunk in async_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/rocket-3B-GGUF',
        model='rocket-3b.Q4_K_M.gguf',
        sample_len=3 * 1024,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and hwo do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 1234 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    print(await async_client.text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/rocket-3B-GGUF',
        model='rocket-3b.Q4_K_M.gguf',
        sample_len=3 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ))

    print(await async_client.chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/rocket-3B-GGUF',
        model='rocket-3b.Q4_K_M.gguf',
        sample_len=3 * 1024,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and hwo do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 1234 simple steps. Explain step by step.'},
        ],
    ))


if __name__ == '__main__':
    # asyncio.run(async_demo_candle_stable_lm())
    asyncio.run(async_demo_llama_cpp_main_stable_lm())
