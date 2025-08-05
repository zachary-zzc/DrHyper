from __future__ import annotations
from config.settings import ConfigManager
from utils.llm_loader import load_chat_model

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"

def format_doctor_response(text: str) -> str:
    """Format doctor's response with color"""
    return f"{Colors.GREEN}{Colors.BOLD}DrHyper:{Colors.RESET} {text}"

def format_patient_input(text: str) -> str:
    """Format patient input with color"""
    return f"{Colors.BLUE}{Colors.BOLD}Patient:{Colors.RESET} {text}"

def format_system_message(text: str) -> str:
    """Format system messages with color"""
    return f"{Colors.YELLOW}System:{Colors.RESET} {text}"

def format_debug(text: str) -> str:
    """Format debug messages with color"""
    return f"{Colors.CYAN}Debug:{Colors.RESET} {text}"

def format_error(text: str) -> str:
    """Format error messages with color"""
    return f"{Colors.RED}Error:{Colors.RESET} {text}"

def parse_json_response(response_content: str) -> dict:
    """
    Parse JSON from response content, handling cases where the JSON might be
    enclosed in markdown code blocks.
    
    Args:
        response_content (str): The response content which may contain JSON
            directly or enclosed in ```json ... ``` code blocks
            
    Returns:
        dict: The parsed JSON data
    """
    import json
    import re
    
    # Check if the content contains markdown JSON code blocks
    json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_block_pattern, response_content)
    
    if match:
        # Extract JSON from inside the code block
        json_content = match.group(1)
    else:
        # Assume the entire content is JSON
        json_content = response_content
    
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}. Content: {json_content[:100]}...")
    
def load_models(verbose=False):
    """Load AI models"""
    print(format_system_message("Loading AI models..."))
    config = ConfigManager()
    try:
        conv_model = load_chat_model(config.conversation_llm.provider, 
                                     config.conversation_llm.model,
                                     api_key=config.conversation_llm.api_key,
                                     base_url=config.conversation_llm.base_url,
                                     model_path=config.conversation_llm.model_path,
                                     max_tokens=config.conversation_llm.max_tokens,
                                     temperature=config.conversation_llm.temperature)
        graph_model = load_chat_model(config.graph_llm.provider,
                                      config.graph_llm.model,
                                      api_key=config.graph_llm.api_key,
                                      base_url=config.graph_llm.base_url,
                                      model_path=config.graph_llm.model_path,
                                      max_tokens=config.graph_llm.max_tokens,
                                      temperature=config.graph_llm.temperature)
        return conv_model, graph_model
    except Exception as e:
        print(format_system_message(f"Error loading models: {e}"))
        if verbose:
            import traceback
            print(format_debug(traceback.format_exc()))
        raise(e)
