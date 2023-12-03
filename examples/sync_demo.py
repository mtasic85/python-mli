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
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
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
        prompt='You are a helpful assistant.\nBuilding a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='candle',
        kind='phi',
        model='1.5',
        model_id='lmz/candle-quantized-phi',
        sample_len=2 * 1024,
        quantized=True,
        stop=['User:'],
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 5 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_stable_lm():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='stable-lm',
        # model_id='lmz/candle-stablelm-3b-4e1t',
        model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=False,
        # use_flash_attn=True,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='candle',
        kind='stable-lm',
        # model_id='lmz/candle-stablelm-3b-4e1t',
        model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=False,
        # use_flash_attn=True,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
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
        sample_len=2 * 1024,
        quantized=True,
        use_flash_attn=False,
        prompt='Building a perfect e-commerce website in 5 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='candle',
        kind='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=True,
        use_flash_attn=False,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 5 simple steps. Explain step by step.'},
        ],
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_quantized_orca_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        kind='quantized',
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        sample_len=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:'
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_llama_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        ctx_size=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_llama_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/Llama-2-7B-Chat-GGUF',
        model='llama-2-7b-chat.Q4_K_M.gguf',
        ctx_size=4 * 1024,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
        messages_syntax='llama',
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        ctx_size=4 * 1024,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
        messages_syntax='llama',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_mistral_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/zephyr-7B-beta-GGUF',
        model='zephyr-7b-beta.Q4_K_M.gguf',
        ctx_size=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)

    print(sync_client.text(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/zephyr-7B-beta-GGUF',
        model='zephyr-7b-beta.Q4_K_M.gguf',
        ctx_size=4 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ))


def sync_demo_llama_cpp_main_mistral_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
        model='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
        ctx_size=4 * 1024,
        stop=['Assistant:', 'User:'],
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
        messages_syntax=None,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/zephyr-7B-beta-GGUF',
        model='zephyr-7b-beta.Q4_K_M.gguf',
        ctx_size=4 * 1024,
        stop=['</s>'],
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'I need help building a website.'},
            {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            {'role': 'user', 'content': 'Building a perfect e-commerce website in 10 simple steps. Explain step by step.'},
        ],
        messages_syntax='zephyr',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_stablelm_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/rocket-3B-GGUF',
        model='rocket-3b.Q4_K_M.gguf',
        ctx_size=3 * 1024,
        prompt='Building a perfect e-commerce website in 1234 simple steps:\nStep 1:',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_stablelm_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/rocket-3B-GGUF',
        model='rocket-3b.Q4_K_M.gguf',
        ctx_size=3 * 1024,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant. You like to ask questions back.'},
            {'role': 'user', 'content': 'Lets have a conversation. I want to know more about you.'},
            # {'role': 'system', 'content': 'You are a helpful assistant.'},
            # {'role': 'user', 'content': 'I need help building a website. Lets have a conversation.'},
            # {'role': 'assistant', 'content': 'Sure, let me know what and how do you need it built.'},
            # {'role': 'user', 'content': 'Building a perfect e-commerce website in 3 simple steps. Explain step by step.'},
        ],
        messages_syntax='chatml',
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_meditron_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        kind='main',
        n_gpu_layers=35,
        model_id='TheBloke/meditron-7B-GGUF',
        model='meditron-7b.Q4_K_M.gguf',
        ctx_size=2 * 1024,
        temp=0.1,
        stop=['<|im_start|>', '<|im_end|>'],
        messages=[
            {'role': 'system', 'content': 'You are an expert medical doctor. You are a gastroenterologist doctor.'},
            {'role': 'question', 'content': 'I need help understanding my condition.'},
            {'role': 'answer', 'content': 'Sure, let me know what do you need to know.'},
            {'role': 'question', 'content': 'What is upper right quadrant endoscopy with bravo? What kind of sensors can be put in patients in gastrointestinal tract?'},
        ],
        messages_syntax='chatml',
    ):
        print(chunk, sep='', end='', flush=True)


if __name__ == '__main__':
    sync_demo_candle_codellama()
    # sync_demo_candle_llama()
    # sync_demo_candle_mistral()
    # sync_demo_candle_phi()
    # sync_demo_candle_phi_quantized()
    # sync_demo_candle_stable_lm()
    # sync_demo_candle_stable_lm_quantized()
    # sync_demo_candle_quantized_orca_text()
    # sync_demo_llama_cpp_main_llama_text()
    # sync_demo_llama_cpp_main_llama_chat()
    # sync_demo_llama_cpp_main_mistral_text()
    # sync_demo_llama_cpp_main_mistral_chat()
    # sync_demo_llama_cpp_main_stablelm_text()
    # sync_demo_llama_cpp_main_stablelm_chat()
    # sync_demo_llama_cpp_main_meditron_chat()
