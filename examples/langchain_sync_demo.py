from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from mli import LangchainMLIClient


def langchain_sync_demo():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LangchainMLIClient(
        endpoint='http://127.0.0.1:5000',
        ws_endpoint='ws://127.0.0.1:5000',
        callback_manager=callback_manager,
        streaming=True,
    )

    engine = 'candle'
    kind = 'stable-lm'
    model_id = 'lmz/candle-stablelm-3b-4e1t'
    
    text = llm(
        'Building a website can be done in 10 simple steps:\nStep 1:',
        engine=engine,
        kind=kind,
        model_id=model_id,
        sample_len=512,
    )
    
    print(text)


if __name__ == '__main__':
    langchain_sync_demo()
