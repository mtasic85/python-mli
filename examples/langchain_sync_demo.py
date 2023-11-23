
def langchain_sync_demo():
    llm = LangchainMLIClient(
        endpoint='http://127.0.0.1:5000',
        ws_endpoint='ws://127.0.0.1:5000',
    )

    text: str = llm('Building a website can be done in 10 simple steps:\nStep 1:')
    print(text)


if __name__ == '__main__':
    langchain_sync_demo()