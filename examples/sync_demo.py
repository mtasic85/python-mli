import os

from mli import SyncMLIClient, format_messages

from defs import (
    ENDPOINT,
    SYSTEM_TEXT,
    PROMPT,
    PROMPT_2,
    ROLE_PROMPT,
    CODE_PROMPT,
    REACT_PROMPT_0,
    REACT_PROMPT_1,
    REACT_PROMPT_2,
    MESSAGES,
    CAR_TEXT,
    CARS_TEXT,
    JSON_FLAT_ARRAY_GRAMMAR,
    JSON_FLAT_OBJECT_GRAMMAR,
)


NGL = os.getenv('NGL')

#
# echo
#
def sync_demo_echo_0():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='echo',
        executable='echo',
        model_id='TheBloke/dolphin-2.6-mistral-7B-GGUF',
        model='dolphin-2.6-mistral-7b.Q4_K_M.gguf',
        creator_model_id='cognitivecomputations/dolphin-2.6-mistral-7b',
        prompt=PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_chat(
        engine='echo',
        executable='echo',
        model_id='TheBloke/dolphin-2.6-mistral-7B-GGUF',
        model='dolphin-2.6-mistral-7b.Q4_K_M.gguf',
        creator_model_id='cognitivecomputations/dolphin-2.6-mistral-7b',
        messages=MESSAGES,
    ):
        print(chunk, sep='', end='', flush=True)
    
    print()


def sync_demo_echo_dolphin_mistral_7b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='echo',
        executable='echo',
        model_id='TheBloke/dolphin-2.6-mistral-7B-GGUF',
        model='dolphin-2.6-mistral-7b.Q4_K_M.gguf',
        creator_model_id='cognitivecomputations/dolphin-2.6-mistral-7b',
        # stop=['<|im_end|>', 'User:', 'Assistant:'],
        stop=['<|im_start|>', '<|im_end|>', 'User:', 'Assistant:'],
        # prompt=PROMPT_2 + '<|im_end|>\n' + '<|im_start|>user\n' + PROMPT_2 + '<|im_end|>\n',
        prompt=PROMPT_2 + '\n<|im_start|>user\n' + PROMPT_2 + '<|im_end|>\n',
    ):
        print(chunk, sep='', end='', flush=True)
    
    print()

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
        prompt=PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_chat(
        engine='candle',
        executable='phi',
        model_id='echo/echo',
        messages=MESSAGES,
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
        prompt=CODE_PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_candle_llama():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='llama',
        model_id='meta-llama/Llama-2-7b-hf',
        sample_len=4 * 1024,
        prompt=PROMPT,
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
        prompt=PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='mistral',
        model_id='lmz/candle-mistral',
        sample_len=4 * 1024,
        quantized=False,
        prompt=PROMPT,
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
        messages=MESSAGES,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='phi',
        model='1.5',
        model_id='Open-Orca/oo-phi-1_5',
        sample_len=2 * 1024,
        quantized=False,
        prompt=PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='phi',
        model='1.5',
        model_id='microsoft/phi-1_5',
        sample_len=2 * 1024,
        quantized=False,
        prompt=PROMPT,
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
        messages=MESSAGES,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='phi',
        model='1.5',
        model_id='lmz/candle-quantized-phi',
        sample_len=2 * 1024,
        quantized=True,
        prompt=PROMPT,
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
        messages=MESSAGES,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='stable-lm',
        model_id='stabilityai/stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=False,
        # use_flash_attn=True,
        prompt=PROMPT,
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
        messages=MESSAGES,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='stable-lm',
        model_id='lmz/candle-stablelm-3b-4e1t',
        sample_len=2 * 1024,
        quantized=True,
        use_flash_attn=False,
        prompt=PROMPT,
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
        messages=MESSAGES,
    ):
        print(chunk, sep='', end='', flush=True)

    for chunk in sync_client.iter_text(
        engine='candle',
        executable='quantized',
        model_id='TheBloke/Orca-2-7B-GGUF',
        model='orca-2-7b.Q4_K_M.gguf',
        sample_len=4 * 1024,
        prompt=PROMPT,
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
        prompt=PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        model_id='echo/echo',
        messages=MESSAGES,
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
        prompt=PROMPT,
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
        messages=MESSAGES,
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
        prompt=PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)

    print(sync_client.text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        prompt=PROMPT,
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
        messages=MESSAGES,
    ):
        print(chunk, sep='', end='', flush=True)

    print()

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/zephyr-7B-beta-GGUF',
        model='zephyr-7b-beta.Q4_K_M.gguf',
        creator_model_id='HuggingFaceH4/zephyr-7b-beta',
        stop=['<|system|>', '<|user|>', '<|assistant|>', '</s>', 'User:', 'Assistant:'],
        messages=MESSAGES,
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
        prompt=PROMPT,
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
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|endoftext|>'],
        messages=MESSAGES,
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
        prompt=PROMPT,
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
        prompt=PROMPT,
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
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|endoftext|>'],
        messages=MESSAGES,
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
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|endoftext|>'],
        messages=MESSAGES,
        prompt_to_file=True,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_yi_9b_200k_text_file():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='LoneStriker/Yi-9B-200K-GGUF',
        model='Yi-9B-200K-Q4_K_M.gguf',
        creator_model_id='01-ai/Yi-9B-200K',
        batch_size=8,
        stop=["<|im_start|>", "<|im_end|>", "User:", "Assistant:"],
        prompt=PROMPT,
        prompt_to_file=True,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_yi_9b_200k_chat_file():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='LoneStriker/Yi-9B-200K-GGUF',
        model='Yi-9B-200K-Q4_K_M.gguf',
        creator_model_id='01-ai/Yi-9B-200K',
        batch_size=8,
        stop=["<|im_start|>", "<|im_end|>", "User:", "Assistant:"],
        messages=MESSAGES,
        prompt_to_file=True,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_yi_6b_200k_text_file():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='LoneStriker/Yi-6B-200K-GGUF',
        model='Yi-6B-200K-Q4_K_M.gguf',
        creator_model_id='01-ai/Yi-6B-200K',
        batch_size=8,
        stop=["User:", "Assistant:"],
        prompt=PROMPT,
        prompt_to_file=True,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_yi_6b_200k_chat_file():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='LoneStriker/Yi-6B-200K-GGUF',
        model='Yi-6B-200K-Q4_K_M.gguf',
        creator_model_id='01-ai/Yi-6B-200K',
        batch_size=8,
        stop=["User:", "Assistant:"],
        messages=MESSAGES,
        prompt_to_file=True,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_qwen1_5_4b_text_file():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='Qwen/Qwen1.5-4B-Chat-GGUF',
        model='qwen1_5-4b-chat-q4_k_m.gguf',
        creator_model_id='Qwen/Qwen1.5-4B-Chat',
        stop=["<|im_start|>", "<|im_end|>", "User:", "Assistant:"],
        prompt=PROMPT,
        prompt_to_file=True,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_qwen1_5_4b_chat_file():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='Qwen/Qwen1.5-4B-Chat-GGUF',
        model='qwen1_5-4b-chat-q4_k_m.gguf',
        creator_model_id='Qwen/Qwen1.5-4B-Chat',
        stop=["<|im_start|>", "<|im_end|>", "User:", "Assistant:"],
        messages=MESSAGES,
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
        model_id='bartowski/gemma-1.1-2b-it-GGUF',
        model='gemma-1.1-2b-it-Q4_K_M.gguf',
        creator_model_id='google/gemma-1.1-2b-it',
        prompt=PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_gemma_2b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        # n_gpu_layers=NGL,
        model_id='bartowski/gemma-1.1-2b-it-GGUF',
        model='gemma-1.1-2b-it-Q4_K_M.gguf',
        creator_model_id='google/gemma-1.1-2b-it',
        stop=['<start_of_turn>', '<end_of_turn>'],
        messages=MESSAGES,
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_gemma_7b_text():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='bartowski/gemma-1.1-7b-it-GGUF',
        model='gemma-1.1-7b-it-Q4_K_M.gguf',
        creator_model_id='google/gemma-1.1-7b-it',
        prompt=PROMPT,
    ):
        print(chunk, sep='', end='', flush=True)


def sync_demo_llama_cpp_main_gemma_7b_chat():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_chat(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='bartowski/gemma-1.1-7b-it-GGUF',
        model='gemma-1.1-7b-it-Q4_K_M.gguf',
        creator_model_id='google/gemma-1.1-7b-it',
        stop=['<start_of_turn>', '<end_of_turn>'],
        messages=MESSAGES,
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

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        # temp=0.1,
        prompt=REACT_PROMPT_0,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_react_1():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='stabilityai/stablelm-2-zephyr-1_6b',
        model='stablelm-2-zephyr-1_6b-Q4_1.gguf',
        creator_model_id='stabilityai/stablelm-2-zephyr-1_6b',
        # temp=0.1,
        prompt=REACT_PROMPT_1,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


def sync_demo_llama_cpp_main_mistral_7b_react_1():
    sync_client = SyncMLIClient(ENDPOINT)

    for chunk in sync_client.iter_text(
        engine='llama.cpp',
        executable='main',
        n_gpu_layers=NGL,
        model_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        creator_model_id='mistralai/Mistral-7B-Instruct-v0.2',
        # temp=0.1,
        prompt=REACT_PROMPT_2,
    ):
        print(chunk, sep='', end='', flush=True)

    print()


if __name__ == '__main__':
    # sync_demo_echo_0()
    # sync_demo_echo_dolphin_mistral_7b_text()

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
    sync_demo_llama_cpp_main_stablelm_zephyr_3b_text()
    sync_demo_llama_cpp_main_stablelm_zephyr_3b_chat()
    sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_text()
    sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_text_file()
    sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_chat()
    sync_demo_llama_cpp_main_stablelm_2_zephyr_1_6b_chat_file()
    # sync_demo_llama_cpp_main_yi_9b_200k_text_file()
    # sync_demo_llama_cpp_main_yi_9b_200k_chat_file()
    # sync_demo_llama_cpp_main_yi_6b_200k_text_file()
    # sync_demo_llama_cpp_main_yi_6b_200k_chat_file()
    # sync_demo_llama_cpp_main_qwen1_5_4b_text_file()
    # sync_demo_llama_cpp_main_qwen1_5_4b_chat_file()
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
    