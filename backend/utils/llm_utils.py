"""LLM 客户端工具：OpenAI 兼容接口，支持 JSON 输出模式与实例缓存。"""

from langchain_openai import ChatOpenAI

from backend.config.lm_config import lm_config

_llm_client_cache: dict[tuple[str, bool], ChatOpenAI] = {}


def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """获取 LLM 客户端单例；model 缺省使用主模型，json_mode 开启 JSON 输出。"""
    model_name = model or lm_config.llm_model
    key = (model_name, json_mode)
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatOpenAI(
        model=model_name,
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        temperature=lm_config.llm_temperature,
        model_kwargs=model_kwargs,
    )
    _llm_client_cache[key] = client
    return client
