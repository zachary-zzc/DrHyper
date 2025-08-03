# config/settings.py
from dataclasses import dataclass
from typing import Dict, Any
import configparser
import os

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    model_path: str = None
    
    max_tokens: int = 8192
    temperature: float = 0.6

@dataclass
class SystemConfig:
    language: str = "English"
    working_directory: str = "./artifacts"
    conversation_directory: str = "./conversations"
    stream: bool = False
    
    # Graph parameters (moved here from GraphConfig)
    node_hit_threshold: float = float('inf')
    confidential_threshold: float = 0.2
    relevance_threshold: float = 0.2
    weight_threshold: float = 0.8
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), 'config.cfg')
        self.config.read(config_path)
        
        # Conversation LLM
        self.conversation_llm = LLMConfig(
            provider=self.config.get('CONVERSATION LLM', 'provider', fallback='openai'),
            model=self.config.get('CONVERSATION LLM', 'model', fallback='gpt-3.5-turbo'),
            api_key=self.config.get('CONVERSATION LLM', 'api_key', fallback=''),
            base_url=self.config.get('CONVERSATION LLM', 'base_url', fallback='https://api.openai.com/v1'),
            model_path=self.config.get('CONVERSATION LLM', 'model_path', fallback=''),
            max_tokens=self.config.getint('CONVERSATION LLM', 'max_tokens', fallback=8192),
            temperature=self.config.getfloat('CONVERSATION LLM', 'temperature', fallback=0.6),
        )
        # Graph LLM
        self.graph_llm = LLMConfig(
            provider=self.config.get('GRAPH LLM', 'provider', fallback='openai'),
            model=self.config.get('GRAPH LLM', 'model', fallback='gpt-3.5-turbo'),
            api_key=self.config.get('GRAPH LLM', 'api_key', fallback=''),
            base_url=self.config.get('GRAPH LLM', 'base_url', fallback='https://api.openai.com/v1'),
            model_path=self.config.get('GRAPH LLM', 'model_path', fallback=''),
            max_tokens=self.config.getint('GRAPH LLM', 'max_tokens', fallback=8192),
            temperature=self.config.getfloat('GRAPH LLM', 'temperature', fallback=0.6),
        )
        # System configuration
        self.system = SystemConfig(
            working_directory=self.config.get('SYSTEM', 'working_directory', fallback="./artifacts"),
            conversation_directory=self.config.get('SYSTEM', 'conversation_directory', fallback="./conversations"),
            language=self.config.get('SYSTEM', 'language', fallback='English'),
            stream=self.config.getboolean('SYSTEM', 'stream', fallback=False),
            # Graph parameters
            node_hit_threshold=self.config.getfloat('GRAPH', 'node_hit_threshold', fallback=float('inf')),
            confidential_threshold=self.config.getfloat('GRAPH', 'confidential_threshold', fallback=0.2),
            relevance_threshold=self.config.getfloat('GRAPH', 'relevance_threshold', fallback=0.2),
            weight_threshold=self.config.getfloat('GRAPH', 'weight_threshold', fallback=0.8),
            alpha=self.config.getfloat('GRAPH', 'alpha', fallback=10.0),
            beta=self.config.getfloat('GRAPH', 'beta', fallback=1.0),
            gamma=self.config.getfloat('GRAPH', 'gamma', fallback=1.0)
        )
        
        self._initialized = True