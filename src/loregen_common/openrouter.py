from pydantic import SecretStr
from typing import Optional
import os
from langchain_openai import ChatOpenAI


class ChatOpenRouter(ChatOpenAI):
    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[SecretStr] = None,
                 openai_api_base: str = 'https://openrouter.ai/api/v1',
                 **kwargs):
        openai_api_key = openai_api_key or SecretStr(os.getenv('OPENROUTER_API_KEY'))
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)
