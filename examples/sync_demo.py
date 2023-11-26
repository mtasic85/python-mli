from mli import SyncMLIClient


ENDPOINT = 'http://127.0.0.1:5000/api/1.0'


def sync_demo_candle_codellama():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='llama',
        model_id='codellama/CodeLlama-7b-Python-hf',
        sample_len=4 * 1024,
        prompt='Lets write python web app for bookstore using aiohttp and pandas. Create dataframes for Users, Books, Rentals, Transactions and Ratings.'
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_llama():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='llama',
        model_id='meta-llama/Llama-2-7b-hf',
        sample_len=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_mistral():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='mistral',
        model_id='lmz/candle-mistral',
        sample_len=4 * 1024,
        quantized=True,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='mistral',
        model_id='lmz/candle-mistral',
        sample_len=4 * 1024,
        quantized=False,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_phi():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='phi',
        model='1.5',
        model_id='Open-Orca/oo-phi-1_5',
        sample_len=2 * 1024,
        quantized=False,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='phi',
        model='1.5',
        model_id='microsoft/phi-1_5',
        sample_len=2 * 1024,
        quantized=False,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='candle',
        kind='phi',
        model='1.5',
        model_id='microsoft/phi-1_5',
        sample_len=2 * 1024,
        quantized=False,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and hwo do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 1234 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_phi_quantized():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='phi',
        model='1.5',
        model_id='lmz/candle-quantized-phi',
        sample_len=2 * 1024,
        quantized=True,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='candle',
        kind='phi',
        model='1.5',
        model_id='lmz/candle-quantized-phi',
        sample_len=2 * 1024,
        quantized=True,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and hwo do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 1234 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_stable_lm():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        # model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=False,
        # use_flash_attn=True,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='candle',
        kind='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        # model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=False,
        # use_flash_attn=True,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and hwo do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 1234 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_stable_lm_quantized():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        # model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=True,
        use_flash_attn=False,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='candle',
        kind='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        # model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=True,
        use_flash_attn=False,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and hwo do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 1234 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_quantized():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='quantized',
        # model='rocket-3b.Q4_K_M.gguf', # bad
        # model='mistral-7b-v0.1.Q4_K_M.gguf', # bad
        # model='yarn-llama-2-7b-128k.Q4_K_M.gguf', # good
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        sample_len=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_llama():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        sample_len=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_mistral():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/zephyr-7B-beta-GGUF',
        model='zephyr-7b-beta.Q4_K_M.gguf',
        sample_len=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    print(sync_client.text(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/zephyr-7B-beta-GGUF',
        model='zephyr-7b-beta.Q4_K_M.gguf',
        sample_len=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ))


def sync_demo_llama_cpp_main_stablelm():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/rocket-3B-GGUF',
        model='rocket-3b.Q4_K_M.gguf',
        sample_len=3 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        kind='main',
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


if __name__ == '__main__':
    # sync_demo_candle_codellama()
    # sync_demo_candle_llama()
    # sync_demo_candle_mistral()
    # sync_demo_candle_phi()
    # sync_demo_candle_phi_quantized()
    # sync_demo_candle_stable_lm()
    # sync_demo_candle_stable_lm_quantized()
    # sync_demo_candle_quantized()
    # sync_demo_llama_cpp_main_llama()
    # sync_demo_llama_cpp_main_mistral()
    sync_demo_llama_cpp_main_stablelm()
