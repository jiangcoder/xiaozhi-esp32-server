from abc import ABC, abstractmethod


class LLMProviderBase(ABC):
    @abstractmethod
    def response(self, session_id, dialogue, headers,opus_base64):
        """LLM response generator"""
        pass
