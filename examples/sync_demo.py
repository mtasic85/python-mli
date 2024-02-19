from mli import SyncMLIClient


ENDPOINT = 'http://127.0.0.1:5000/api/1.0'
SYSTEM_TEXT = 'You are an intelligent, helpful, respectful and honest assistant.'
STABLE_LM_SYSTEM_TEXT = 'You are a helpful assistant.'
MD_SYSTEM_TEXT = 'You are an intelligent, helpful, respectful and honest Doctor of Medicine.'


#
# candle
#
def sync_demo_candle_codellama():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='llama',
        model_id='codellama/CodeLlama-7b-Python-hf',
        sample_len=4 * 1024,
        prompt='Lets write python web app for bookstore using aiohttp and pandas. Create dataframes for Users, Books, Rentals, Transactions and Ratings.'
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_llama():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='llama',
        model_id='meta-llama/Llama-2-7b-hf',
        sample_len=4 * 1024,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_mistral():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='mistral',
        model_id='lmz/candle-mistral',
        sample_len=4 * 1024,
        quantized=True,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='mistral',
        model_id='lmz/candle-mistral',
        sample_len=4 * 1024,
        quantized=False,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_phi():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='candle',
        executable='phi',
        model='1.5',
        model_id='microsoft/phi-1_5',
        sample_len=2 * 1024,
        quantized=False,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 5 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='phi',
        model='1.5',
        model_id='Open-Orca/oo-phi-1_5',
        sample_len=2 * 1024,
        quantized=False,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='phi',
        model='1.5',
        model_id='microsoft/phi-1_5',
        sample_len=2 * 1024,
        quantized=False,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_phi_quantized():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='candle',
        executable='phi',
        model='1.5',
        model_id='lmz/candle-quantized-phi',
        sample_len=2 * 1024,
        quantized=True,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 5 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='phi',
        model='1.5',
        model_id='lmz/candle-quantized-phi',
        sample_len=2 * 1024,
        quantized=True,
        prompt=f'{SYSTEM_TEXT}\nBuilding a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_stable_lm():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='candle',
        executable='stable-lm',
        model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=False,
        # use_flash_attn=True,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': STABLE_LM_SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Explain building e-commerce website in 5 steps.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='stable-lm',
        model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=False,
        # use_flash_attn=True,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_stable_lm_quantized():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='candle',
        executable='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=True,
        use_flash_attn=False,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': STABLE_LM_SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 5 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=True,
        use_flash_attn=False,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_quantized_orca_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='candle',
        executable='quantized',
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        sample_len=2 * 1024,
        quantized=True,
        # use_flash_attn=False,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 5 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='quantized',
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        sample_len=4 * 1024,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)


#
# llama.cpp
#
def sync_demo_llama_cpp_main_orca2_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        creator_model_id='microsoft/Orca-2-7b',
        ctx_size=4 * 1024,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_orca2_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        creator_model_id='microsoft/Orca-2-7b',
        ctx_size=4 * 1024,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
        # messages_syntax='llama',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_mistral_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        ctx_size=4 * 1024,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    print(sync_client.text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        ctx_size=4 * 1024,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ))


def sync_demo_llama_cpp_main_mistral_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        ctx_size=4 * 1024,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
        messages_syntax=None,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/zephyr-7B-beta-GGUF',
        model='zephyr-7b-beta.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-v0.2',
        ctx_size=4 * 1024,
        stop=['</s>'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
        messages_syntax='zephyr',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_stablelm_zephyr_3b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/stablelm-zephyr-3b-GGUF',
        model='stablelm-zephyr-3b.Q4_K_M.gguf',
        creator_model_id='stabilityai/stablelm-zephyr-3b',
        ctx_size=3 * 1024,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_stablelm_zephyr_3b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=35,
        model_id='TheBloke/stablelm-zephyr-3b-GGUF',
        model='stablelm-zephyr-3b.Q4_K_M.gguf',
        creator_model_id='stabilityai/stablelm-zephyr-3b',
        ctx_size=3 * 1024,
        stop=["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"],
        messages=[
            {'role': 'system', 'content': f'{SYSTEM_TEXT}. You like to ask questions back.'},
            {'role': 'user', 'content': 'Lets have a conversation. I want to know more about you.'},
        ],
        messages_syntax='chatml',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        # n_gpu_layers=35,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        ctx_size=2 * 1024,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        # n_gpu_layers=35,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        ctx_size=2 * 1024,
        stop=["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"],
        messages=[
            {'role': 'system', 'content': f'{SYSTEM_TEXT}. You like to ask questions back.'},
            {'role': 'user', 'content': 'Lets have a conversation. I want to know more about you.'},
        ],
        messages_syntax='stablelm-2-zephyr-1_6b',
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
    # sync_demo_candle_quantized_orca_text()
    
    # sync_demo_llama_cpp_main_orca2_text()
    # sync_demo_llama_cpp_main_orca2_chat()
    # sync_demo_llama_cpp_main_mistral_text()
    # sync_demo_llama_cpp_main_mistral_chat()
    # sync_demo_llama_cpp_main_stablelm_zephyr_3b_text()
    # sync_demo_llama_cpp_main_stablelm_zephyr_3b_chat()
    sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_text()
    sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_chat()
