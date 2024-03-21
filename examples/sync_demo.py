import os

from mli import SyncMLIClient, format_messages

from defs import (
    ENDPOINT,
    SYSTEM_TEXT,
    CAR_TEXT,
    CARS_TEXT,
    JSON_FLAT_ARRAY_GRAMMAR,
    JSON_FLAT_OBJECT_GRAMMAR,
)


NGL = os.getenv('NGL')


#
# candle
#
def sync_demo_candle_echo():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='llama',
        model_id='echo/echo',
        sample_len=4 * 1024,
        prompt='Lets write python web app for bookstore using aiohttp and pandas. Create dataframes for Users, Books, Rentals, Transactions and Ratings.'
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_chat(
        engine='candle',
        executable='phi',
        model_id='echo/echo',
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 5 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)
    
    print()


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
            {'role': 'system', 'content': SYSTEM_TEXT},
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
            {'role': 'system', 'content': SYSTEM_TEXT},
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
def sync_demo_llama_cpp_main_echo():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        model_id='echo/echo',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        model_id='echo/echo',
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)
    
    print()


def sync_demo_llama_cpp_main_orca2_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        creator_model_id='microsoft/Orca-2-7b',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_orca2_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        creator_model_id='microsoft/Orca-2-7b',
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_mistral_7b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    print(sync_client.text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ))


def sync_demo_llama_cpp_main_mistral_7b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/zephyr-7B-beta-GGUF',
        model='zephyr-7b-beta.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-v0.2',
        stop=['</s>'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_mistrallite_7b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='second-state/MistralLite-7B-GGUF',
        model='MistralLite-Q4_K_M.gguf',
        creator_model_id='amazon/MistralLite',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    print(sync_client.text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='second-state/MistralLite-7B-GGUF',
        model='MistralLite-Q4_K_M.gguf',
        creator_model_id='amazon/MistralLite',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ))


def sync_demo_llama_cpp_main_mistrallite_7b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='second-state/MistralLite-7B-GGUF',
        model='MistralLite-Q4_K_M.gguf',
        creator_model_id='amazon/MistralLite',
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='second-state/MistralLite-7B-GGUF',
        model='MistralLite-Q4_K_M.gguf',
        creator_model_id='amazon/MistralLite',
        stop=['</s>'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_zephyr_3b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/stablelm-zephyr-3b-GGUF',
        model='stablelm-zephyr-3b.Q4_K_M.gguf',
        creator_model_id='stabilityai/stablelm-zephyr-3b',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_zephyr_3b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/stablelm-zephyr-3b-GGUF',
        model='stablelm-zephyr-3b.Q4_K_M.gguf',
        creator_model_id='stabilityai/stablelm-zephyr-3b',
        stop=["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"],
        messages=[
            {'role': 'system', 'content': f'{SYSTEM_TEXT}. You like to ask questions back.'},
            {'role': 'user', 'content': 'Lets have a conversation. I want to know more about you.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_text_file():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
        prompt_to_file=True,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        stop=["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"],
        messages=[
            {'role': 'system', 'content': f'{SYSTEM_TEXT}. You like to ask questions back.'},
            {'role': 'user', 'content': 'Lets have a conversation. I want to know more about you.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_chat_file():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        stop=["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"],
        messages=[
            {'role': 'system', 'content': f'{SYSTEM_TEXT}. You like to ask questions back.'},
            {'role': 'user', 'content': 'Lets have a conversation. I want to know more about you.'},
        ],
        prompt_to_file=True,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_gemma_2b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='second-state/Gemma-2b-it-GGUF',
        model='gemma-2b-it-Q4_K_M.gguf',
        creator_model_id='google/gemma-2b-it',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_gemma_2b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='second-state/Gemma-2b-it-GGUF',
        model='gemma-2b-it-Q4_K_M.gguf',
        creator_model_id='google/gemma-2b-it',
        stop=['<start_of_turn>', '<end_of_turn>'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_gemma_7b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='second-state/Gemma-7b-it-GGUF',
        model='gemma-7b-it-Q4_K_M.gguf',
        creator_model_id='google/gemma-7b-it',
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_gemma_7b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='second-state/Gemma-7b-it-GGUF',
        model='gemma-7b-it-Q4_K_M.gguf',
        creator_model_id='google/gemma-7b-it',
        stop=['<start_of_turn>', '<end_of_turn>'],
        messages=[
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


#
# grammar
#
def sync_demo_llama_cpp_main_mistral_7b_grammar():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        temp=0.1,
        top_k=100,
        top_p=0.95,
        prompt=f'Parse text as JSON object:\n```{CAR_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_OBJECT_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        temp=0.1,
        top_k=100,
        top_p=0.95,
        prompt=f'Parse text as JSON array of objects:\n```{CARS_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_ARRAY_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_grammar():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        temp=0.8,
        top_k=200,
        top_p=0.8,
        prompt=f'Parse text as JSON object:\n```{CAR_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_OBJECT_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        temp=0.8,
        top_k=200,
        top_p=0.8,
        prompt=f'Parse text as JSON array of objects:\n```{CAR_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_ARRAY_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        temp=0.8,
        top_k=200,
        top_p=0.8,
        prompt=f'Parse text as JSON array of objects:\n```{CARS_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_ARRAY_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stable_code_3b_grammar():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/stable-code-3b-GGUF',
        model='stable-code-3b.Q4_K_M.gguf',
        creator_model_id='stabilityai/stable-code-3b',
        temp=0.1,
        top_k=200,
        prompt=f'Parse text as JSON object:\n```{CAR_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_OBJECT_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/stable-code-3b-GGUF',
        model='stable-code-3b.Q4_K_M.gguf',
        creator_model_id='stabilityai/stable-code-3b',
        temp=0.1,
        top_k=200,
        prompt=f'Parse text as JSON array of objects:\n```{CARS_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_ARRAY_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_qwen1_5_0_5b_chat_grammar():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='Qwen/Qwen1.5-0.5B-Chat-GGUF',
        model='qwen1_5-0_5b-chat-q4_k_m.gguf',
        creator_model_id='Qwen/Qwen1.5-0.5B-Chat',
        # temp=0.5,
        # top_k=200,
        prompt=f'Parse text as JSON object:\n```{CAR_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_OBJECT_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='Qwen/Qwen1.5-0.5B-Chat-GGUF',
        model='qwen1_5-0_5b-chat-q4_k_m.gguf',
        creator_model_id='Qwen/Qwen1.5-0.5B-Chat',
        # temp=0.5,
        # top_k=200,
        prompt=f'Parse text as JSON array of objects:\n```{CARS_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_ARRAY_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_phi_1_5_grammar():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TKDKid1000/phi-1_5-GGUF',
        model='phi-1_5-Q4_K_M.gguf',
        creator_model_id='microsoft/phi-1_5',
        # temp=0.1,
        # top_k=10,
        prompt=f'Parse text as JSON object:\n```{CAR_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_OBJECT_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TKDKid1000/phi-1_5-GGUF',
        model='phi-1_5-Q4_K_M.gguf',
        creator_model_id='microsoft/phi-1_5',
        # temp=0.1,
        # top_k=10,
        prompt=f'Parse text as JSON array of objects:\n```{CARS_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_ARRAY_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_phi_2_0_grammar():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/phi-2-GGUF',
        model='phi-2.Q4_K_M.gguf',
        creator_model_id='microsoft/phi-2',
        # temp=0.1,
        # top_k=10,
        prompt=f'Parse text as JSON object:\n```{CAR_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_OBJECT_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/phi-2-GGUF',
        model='phi-2.Q4_K_M.gguf',
        creator_model_id='microsoft/phi-2',
        # temp=0.1,
        # top_k=10,
        prompt=f'Parse text as JSON array of objects:\n```{CARS_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_ARRAY_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_tinyllama_1_1b_chat_v1_0_grammar():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
        model='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        creator_model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        # temp=0.1,
        # top_k=200,
        prompt=f'Parse text as JSON object:\n```{CAR_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_OBJECT_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
        model='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        creator_model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        # temp=0.1,
        # top_k=200,
        prompt=f'Parse text as JSON array of objects:\n```{CARS_TEXT}```\nJSON: ',
        grammar=JSON_FLAT_ARRAY_GRAMMAR,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


#
# react
#
def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_react_0():
    sync_client = SyncMLIClient(ENDPOINT)

    prompt = '''You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 1, a coffee machine 1, a countertop 3, a stove burner 1, and a toaster 1.
Your task is to: Put some pepper shaker in/on a drawer 1.

Act 1: Think.

Act 2: Go on cabinet 1.
Obs 2: On the cabinet 1, you see a vase 2.

Act 6: Go on countertop 3.
Obs 6: On the countertop 3, you see an apple 1, a bread 1, a pepper shaker 1, and a vase 3.

Act 7: Take pepper shaker 1 from countertop 3.
Obs 7: You pick up the pepper shaker 1 from the countertop 3.

Act 8: Think.

Act 9: Go to drawer 1.
Obs 9: Drawer 1 is closed.

Act 10: Open drawer 1.
Obs 10: You opened drawer 1.

Act 11: Put pepper shaker 1 in/on drawer 1.
Obs 11:'''

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        # temp=0.1,
        prompt=prompt,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_react_1():
    sync_client = SyncMLIClient(ENDPOINT)

    question = 'Who is and how old is current president of the Republic of Serbia, but have in mind that current year is 2024?'

    prompt = f'''Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: Lookup[eastern sector]
Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action: Search[High Plains]
Observation: High Plains refers to one of two distinct land regions
Thought: I need to instead search High Plains (United States).
Action: Search[High Plains (United States)]
Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action: Search[Nicholas Ray]
Observation: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action: Search[Elia Kazan]
Observation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur’s Magazine or First for Women?
Thought: I need to search Arthur’s Magazine and First for Women, and find which was started first.
Action: Search[Arthur’s Magazine]
Observation: Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought: Arthur’s Magazine was started in 1844. I need to search First for Women next.
Action: Search[First for Women]
Observation: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.
Action: Finish[Arthur’s Magazine]

Question: {question}'''

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        # temp=0.1,
        prompt=prompt,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_mistral_7b_react_1():
    sync_client = SyncMLIClient(ENDPOINT)

    question = 'Who is and how old is current president of the Republic of Serbia, but have in mind that current year is 2024?'

    prompt = f'''Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: Lookup[eastern sector]
Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action: Search[High Plains]
Observation: High Plains refers to one of two distinct land regions
Thought: I need to instead search High Plains (United States).
Action: Search[High Plains (United States)]
Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action: Search[Nicholas Ray]
Observation: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action: Search[Elia Kazan]
Observation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur’s Magazine or First for Women?
Thought: I need to search Arthur’s Magazine and First for Women, and find which was started first.
Action: Search[Arthur’s Magazine]
Observation: Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought: Arthur’s Magazine was started in 1844. I need to search First for Women next.
Action: Search[First for Women]
Observation: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.
Action: Finish[Arthur’s Magazine]

Question: {question}'''

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        # temp=0.1,
        prompt=prompt,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


if __name__ == '__main__':
    # sync_demo_candle_echo()
    # sync_demo_llama_cpp_main_echo()

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
    # sync_demo_llama_cpp_main_mistral_7b_text()
    # sync_demo_llama_cpp_main_mistral_7b_chat()
    # sync_demo_llama_cpp_main_mistrallite_7b_text()
    # sync_demo_llama_cpp_main_mistrallite_7b_chat()
    # sync_demo_llama_cpp_main_stablelm_zephyr_3b_text()
    # sync_demo_llama_cpp_main_stablelm_zephyr_3b_chat()
    # sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_text()
    # sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_text_file()
    # sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_chat()
    sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_chat_file()
    # sync_demo_llama_cpp_main_gemma_2b_text()
    # sync_demo_llama_cpp_main_gemma_2b_chat()
    # sync_demo_llama_cpp_main_gemma_7b_text()
    # sync_demo_llama_cpp_main_gemma_7b_chat()

    # sync_demo_llama_cpp_main_mistral_7b_grammar()
    # sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_grammar()
    # sync_demo_llama_cpp_main_stable_code_3b_grammar()
    # sync_demo_llama_cpp_main_qwen1_5_0_5b_chat_grammar()
    # sync_demo_llama_cpp_main_phi_1_5_grammar()
    # sync_demo_llama_cpp_main_phi_2_0_grammar()
    # sync_demo_llama_cpp_main_tinyllama_1_1b_chat_v1_0_grammar()

    # sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_react_0()
    # sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_react_1()
    # sync_demo_llama_cpp_main_mistral_7b_react_1()
    