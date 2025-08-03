# base.py
from typing import Dict, Any, List
from string import Template
from config.settings import ConfigManager

class BasePrompt:
    """Base class for prompt management"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.prompt_templates = {}  # Will be populated by child classes
    
    def get(self, prompt_name: str, default: str = None, **kwargs) -> str:
        """
        Get prompt template with variable substitutions using string.Template
        
        Args:
            prompt_name: Name of the prompt to retrieve
            default: Default value to return if prompt doesn't exist
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string or default value if prompt doesn't exist
        """
        if prompt_name not in self.prompt_templates:
            if default is not None:
                return default
            raise ValueError(f"Prompt '{prompt_name}' not found. ")
        
        # Use safe_substitute to avoid KeyError for missing variables
        return Template(self.prompt_templates[prompt_name]).safe_substitute(**kwargs)