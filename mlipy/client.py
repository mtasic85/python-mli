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
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema.output import GenerationChunk

from .server import LLMParams


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


class LangchainMLIClient(LLM):
    endpoint: str = 'http://127.0.0.1:5000'
    ws_endpoint: str = 'ws://127.0.0.1:5000'
    verbose: bool = False

    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'LangchainMLIClient'

    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            'endpoint': self.endpoint,
            'ws_endpoint': self.ws_endpoint,
        }


    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Unpack[LLMParams],
    ) -> str:
        sync_client = SyncMLIClient(self.endpoint, self.ws_endpoint)
        res: dict = sync_client.text(prompt=prompt, stop=stop, **kwargs)
        output: str = res['output']
        logprobs = None

        if run_manager:
            run_manager.on_llm_new_token(
                token=output,
                verbose=self.verbose,
                log_probs=logprobs,
            )

        return output


    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Unpack[LLMParams],
    ) -> str:
        """Run the LLM on the given prompt and input."""
        async_client = AsyncMLIClient(self.endpoint, self.ws_endpoint)
        res: dict = await async_client.text(prompt=prompt, stop=stop, **kwargs)
        output: str = res['output']
        logprobs = None

        if run_manager:
            await run_manager.on_llm_new_token(
                token=output,
                verbose=self.verbose,
                log_probs=logprobs,
            )

        return output


    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Unpack[LLMParams],
    ) -> Iterator[GenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the LLM class method of the same name.
        """
        sync_client = SyncMLIClient(self.endpoint, self.ws_endpoint)
        logprobs = None

        for text in sync_client.iter_text(prompt=prompt, stop=stop, **kwargs):
            chunk = GenerationChunk(
                text=text,
                generation_info={'logprobs': logprobs},
            )

            yield chunk

            if run_manager:
                run_manager.on_llm_new_token(
                    token=chunk.text,
                    verbose=self.verbose,
                    log_probs=logprobs,
                )


    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Unpack[LLMParams],
    ) -> AsyncIterator[GenerationChunk]:
        async_client = AsyncMLIClient(self.endpoint, self.ws_endpoint)
        logprobs = None

        async for text in async_client.iter_text(prompt=prompt, stop=stop, **kwargs):
            chunk = GenerationChunk(
                text=text,
                generation_info={'logprobs': logprobs},
            )

            yield chunk

            if run_manager:
                await run_manager.on_llm_new_token(
                    token=chunk.text,
                    verbose=self.verbose,
                    log_probs=logprobs,
                )
