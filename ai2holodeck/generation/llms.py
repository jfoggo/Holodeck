from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from ai2holodeck.constants import (
    LLM_MODEL_NAME_OPENAI,
    LLM_MODEL_NAME_GOOGLEAI,
)

class LLM:
    """Wrapper class to allow support of multiple providers"""

    def __init__(
        self,
        provider: Literal["openai", "googleai"] = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: int = 2048,
    ):
        self.provider = provider

        if self.provider == "openai":
            if model_name is None: model_name = LLM_MODEL_NAME_OPENAI

            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                max_tokens=max_tokens,
            )

        elif self.provider == "googleai":
            if model_name is None: model_name = LLM_MODEL_NAME_GOOGLEAI

            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                max_tokens=max_tokens,
            )
        
        else:
            raise Exception(f"Unexpected provider: {self.provider}")
    
    def __call__(
        self,
        prompt: str
    ) -> str:
        """Let the LLM generate a response based on the given input prompt."""
        
        if self.provider == "openai":
            return self.llm.invoke(prompt).content
        
        elif self.provider == "googleai":
            return self.llm.invoke(prompt).content
        
        else:
            raise Exception(f"Unexpected provider: {self.provider}")