from abc import ABC, abstractmethod


class LLMProviderBase(ABC):
    @abstractmethod
    def response(self, session_id, dialogue, headers):
        """LLM response generator"""
        pass
