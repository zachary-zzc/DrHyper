def parse_json_response(response_content):
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
