__all__ = ['SyncMLIClient', 'AsyncMLIClient']

import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import json
from typing import Iterator, AsyncIterator

from aiohttp import ClientSession, WSMsgType


class BaseMLIClient:
    endpoint: str
    ws_endpoint: str


    def __init__(self, endpoint: str, ws_endpoint: str):
        self.endpoint = endpoint
        self.ws_endpoint = ws_endpoint


class SyncMLIClient(BaseMLIClient):
    def __init__(self, endpoint: str, ws_endpoint: str):
        super().__init__(endpoint, ws_endpoint)
        self.async_client = AsyncMLIClient(endpoint, ws_endpoint)


    def text(self, **kwargs) -> str:
        data = asyncio.run(self.async_client.text(**kwargs))
        return data


    def chat(self, **kwargs) -> str:
        data = asyncio.run(self.async_client.chat(**kwargs))
        return data


    def iter_text(self, **kwargs) -> Iterator[str]:
        # for chunk in asyncio.run(self.async_client.iter_text(**kwargs)):
        #     yield chunk
        raise NotImplementedError


    def iter_chat(self, **kwargs) -> Iterator[str]:
        # for chunk in asyncio.run(self.async_client.iter_chat(**kwargs)):
        #     yield chunk
        raise NotImplementedError


class AsyncMLIClient(BaseMLIClient):
    async def text(self, **kwargs) -> str:
        url: str = f'{self.endpoint}/api/1.0/text/completions'

        async with ClientSession() as session:
            async with session.post(url, json=kwargs) as resp:
                data = await resp.json()

        return data


    async def chat(self, **kwargs) -> str:
        url: str = f'{self.endpoint}/api/1.0/chat/completions'

        async with ClientSession() as session:
            async with session.post(url, json=kwargs) as resp:
                data = await resp.json()

        return data


    async def iter_text(self, **kwargs) -> AsyncIterator[str]:
        url: str = f'{self.ws_endpoint}/api/1.0/text/completions'
        
        async with ClientSession() as session:
            async with session.ws_connect(url) as ws:
                await ws.send_json(kwargs)

                async for msg in ws:
                    if msg.type == WSMsgType.PING:
                        await ws.pong(msg.data)
                    elif msg.type == WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        yield data['chunk']
                    elif msg.type == WSMsgType.ERROR:
                        print(f'[ERROR] websocket closed with exception: {ws.exception()}')
                        break


    async def iter_chat(self, **kwargs) -> AsyncIterator[str]:
        url: str = f'{self.ws_endpoint}/api/1.0/chat/completions'
        
        async with ClientSession() as session:
            async with session.ws_connect(url) as ws:
                await ws.send_json(kwargs)

                async for msg in ws:
                    if msg.type == WSMsgType.PING:
                        await ws.pong(msg.data)
                    elif msg.type == WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        yield data['chunk']
                    elif msg.type == WSMsgType.ERROR:
                        print(f'[ERROR] websocket closed with exception: {ws.exception()}')
                        break


def sync_demo():
    sync_client = SyncMLIClient('http://127.0.0.1:5000', 'ws://127.0.0.1:5000')

    print(sync_client.text(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "prompt": "Building a website can be done in 10 simple steps:\nStep 1:"
    }))

    print(sync_client.chat(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I need help building a website."},
            {"role": "assistant", "content": "Sure, let me know what and hwo do you need it built."},
            {"role": "user", "content": "Building a website can be done in 10 simple steps. Explain step by step."}
        ]
    }))

    # for chunk in sync_client.iter_text(**{
    #     "engine": "candle",
    #     "kind": "stable-lm",
    #     "model_id": "lmz/candle-stablelm-3b-4e1t",
    #     "prompt": "Building a website can be done in 10 simple steps:\nStep 1:"
    # }):
    #     print(chunk)

    # for chunk in sync_client.iter_chat(**{
    #     "engine": "candle",
    #     "kind": "stable-lm",
    #     "model_id": "lmz/candle-stablelm-3b-4e1t",
    #     "messages": [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "I need help building a website."},
    #         {"role": "assistant", "content": "Sure, let me know what and hwo do you need it built."},
    #         {"role": "user", "content": "Building a website can be done in 10 simple steps. Explain step by step."}
    #     ]
    # }):
    #     print(chunk)


async def async_demo():
    async_client = AsyncMLIClient('http://127.0.0.1:5000', 'ws://127.0.0.1:5000')

    print(await async_client.text(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "prompt": "Building a website can be done in 10 simple steps:\nStep 1:"
    }))

    print(await async_client.chat(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
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
        "prompt": "Building a website can be done in 10 simple steps:\nStep 1:"
    }):
        print(chunk)

    async for chunk in async_client.iter_chat(**{
        "engine": "candle",
        "kind": "stable-lm",
        "model_id": "lmz/candle-stablelm-3b-4e1t",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I need help building a website."},
            {"role": "assistant", "content": "Sure, let me know what and hwo do you need it built."},
            {"role": "user", "content": "Building a website can be done in 10 simple steps. Explain step by step."}
        ]
    }):
        print(chunk)


if __name__ == '__main__':
    sync_demo()
    asyncio.run(async_demo())
