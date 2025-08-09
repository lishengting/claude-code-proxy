import asyncio
import json
import time
from datetime import datetime
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError
from src.core.logging import logger
from src.core.usage_stats import append_usage_tsv

class OpenAIClient:
    """Async OpenAI client with cancellation support."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 90, api_version: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        
        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout
            )
            self.api_type = "azure"
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
            self.api_type = "openai"
        self.active_requests: Dict[str, asyncio.Event] = {}
    
    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""
        # 进入日志（功能函数）
        logger.info(
            f"[ENTER] create_chat_completion: request_id={request_id}, model={request.get('model')}, base_url={self.base_url}"
        )
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            start_ts = time.perf_counter()
            start_time_str = datetime.utcnow().isoformat()
            # Create task that can be cancelled
            completion_task = asyncio.create_task(
                self.client.chat.completions.create(**request)
            )
            
            if request_id:
                # Wait for either completion or cancellation
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check if request was cancelled
                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(status_code=499, detail="Request cancelled by client")
                
                completion = await completion_task
            else:
                completion = await completion_task
            
            # Convert to dict format that matches the original interface
            # [删除替换-原因] 原先直接返回 completion.model_dump()；为记录usage与写TSV，改为保存为 result_dict 并在记录后返回
            # return completion.model_dump()
            result_dict = completion.model_dump()
            # 提取usage
            usage = result_dict.get("usage", {}) or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cache_read_input_tokens = 0
            prompt_tokens_details = usage.get("prompt_tokens_details", {}) or {}
            if prompt_tokens_details:
                cache_read_input_tokens = prompt_tokens_details.get("cached_tokens", 0)
            latency_ms = int((time.perf_counter() - start_ts) * 1000)

            # 成功日志（功能函数）
            logger.info(
                f"[SUCCESS] create_chat_completion: request_id={request_id}, model={request.get('model')}, in={input_tokens}, out={output_tokens}, cache_in={cache_read_input_tokens}, latency_ms={latency_ms}"
            )
            # 写TSV
            append_usage_tsv(
                {
                    "timestamp": start_time_str,
                    "request_id": request_id or "",
                    "is_stream": False,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": cache_read_input_tokens,
                    "total_tokens": (input_tokens or 0) + (output_tokens or 0),
                    "latency_ms": latency_ms,
                    "status": "success",
                    "error": "",
                }
            )
            return result_dict
        
        except AuthenticationError as e:
            # 失败日志（功能函数）
            logger.error(f"[FAIL] create_chat_completion auth error: request_id={request_id}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            # 写TSV
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": False,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"auth:{str(e)}",
                }
            )
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            logger.error(f"[FAIL] create_chat_completion rate limit: request_id={request_id}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": False,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"ratelimit:{str(e)}",
                }
            )
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            logger.error(f"[FAIL] create_chat_completion bad request: request_id={request_id}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": False,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"badrequest:{str(e)}",
                }
            )
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            logger.error(f"[FAIL] create_chat_completion api error: request_id={request_id}, status={status_code}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": False,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"api:{str(e)}",
                }
            )
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            logger.error(f"[FAIL] create_chat_completion unexpected: request_id={request_id}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": False,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"unexpected:{str(e)}",
                }
            )
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def create_chat_completion_stream(self, request: Dict[str, Any], request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""
        # 进入日志（功能函数）
        logger.info(
            f"[ENTER] create_chat_completion_stream: request_id={request_id}, model={request.get('model')}, base_url={self.base_url}"
        )
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            start_ts = time.perf_counter()
            start_time_str = datetime.utcnow().isoformat()
            last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_tokens_details": {}}
            # Ensure stream is enabled
            request["stream"] = True
            if "stream_options" not in request:
                request["stream_options"] = {}
            request["stream_options"]["include_usage"] = True
            
            # Create the streaming completion
            streaming_completion = await self.client.chat.completions.create(**request)
            
            async for chunk in streaming_completion:
                # Check for cancellation before yielding each chunk
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(status_code=499, detail="Request cancelled by client")
                
                # Convert chunk to SSE format matching original HTTP client format
                chunk_dict = chunk.model_dump()
                # 跟踪用量
                usage = chunk_dict.get("usage")
                if usage:
                    last_usage = usage
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"
            
            # Signal end of stream
            yield "data: [DONE]"
            # 结束时记录成功日志与TSV
            input_tokens = (last_usage or {}).get("prompt_tokens", 0)
            output_tokens = (last_usage or {}).get("completion_tokens", 0)
            cache_read_input_tokens = 0
            prompt_tokens_details = (last_usage or {}).get("prompt_tokens_details", {}) or {}
            if prompt_tokens_details:
                cache_read_input_tokens = prompt_tokens_details.get("cached_tokens", 0)
            latency_ms = int((time.perf_counter() - start_ts) * 1000)
            logger.info(
                f"[SUCCESS] create_chat_completion_stream: request_id={request_id}, model={request.get('model')}, in={input_tokens}, out={output_tokens}, cache_in={cache_read_input_tokens}, latency_ms={latency_ms}"
            )
            append_usage_tsv(
                {
                    "timestamp": start_time_str,
                    "request_id": request_id or "",
                    "is_stream": True,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": cache_read_input_tokens,
                    "total_tokens": (input_tokens or 0) + (output_tokens or 0),
                    "latency_ms": latency_ms,
                    "status": "success",
                    "error": "",
                }
            )
                
        except AuthenticationError as e:
            logger.error(f"[FAIL] create_chat_completion_stream auth error: request_id={request_id}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": True,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"auth:{str(e)}",
                }
            )
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            logger.error(f"[FAIL] create_chat_completion_stream rate limit: request_id={request_id}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": True,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"ratelimit:{str(e)}",
                }
            )
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            logger.error(f"[FAIL] create_chat_completion_stream bad request: request_id={request_id}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": True,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"badrequest:{str(e)}",
                }
            )
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            logger.error(f"[FAIL] create_chat_completion_stream api error: request_id={request_id}, status={status_code}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": True,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"api:{str(e)}",
                }
            )
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            logger.error(f"[FAIL] create_chat_completion_stream unexpected: request_id={request_id}, err={e}")
            latency_ms = 0
            try:
                latency_ms = int((time.perf_counter() - start_ts) * 1000)
            except Exception:
                pass
            append_usage_tsv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id or "",
                    "is_stream": True,
                    "model": request.get("model", ""),
                    "base_url": self.base_url,
                    "api_type": self.api_type,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"unexpected:{str(e)}",
                }
            )
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."
        
        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False