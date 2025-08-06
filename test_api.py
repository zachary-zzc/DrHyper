import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Initialize conversation
def init_conversation(name, age, gender):
    init_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "model": "DrHyper"
    }
    response = requests.post(f"{BASE_URL}/init_conversation", json=init_data)
    if response.status_code != 200:
        raise Exception(f"Failed to initialize: {response.json()['detail']}")
    return response.json()

# Send a message and get a response
def send_message(conversation_id, message):
    chat_data = {
        "conversation_id": conversation_id,
        "human_message": message
    }
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    if response.status_code != 200:
        raise Exception(f"Failed to send message: {response.json()['detail']}")
    return response.json()

# End the conversation
def end_conversation(conversation_id):
    end_data = {
        "conversation_id": conversation_id
    }
    response = requests.post(f"{BASE_URL}/end_conversation", json=end_data)
    if response.status_code != 200:
        raise Exception(f"Failed to end conversation: {response.json()['detail']}")
    return response.json()

# Main execution flow
try:
    # Start conversation
    print("starting conversation...")
    result = init_conversation("Li Ming", 62, "male")
    conversation_id = result["conversation_id"]
    print(f"Conversation started with ID: {conversation_id}")
    print(f"AI: {result['ai_message']}")
    
    # Sample conversation flow
    messages = [
        "My blood pressure has been high lately, around 150/95",
        "I've been feeling dizzy in the mornings",
        "Yes, I take medication but sometimes forget",
        "I don't exercise much and eat a lot of salty food"
    ]
    
    for message in messages:
        print(f"\nPatient: {message}")
        response = send_message(conversation_id, message)
        print(f"AI: {response['ai_message']}")
        
        # Check if diagnosis is complete
        if response["accomplish"]:
            print("\nDiagnosis completed automatically.")
            break
    
    # Manually end the conversation if not already completed
    if not response.get("accomplish", False):
        end_result = end_conversation(conversation_id)
        print(f"\nConversation ended: {end_result['conversation_id']}")

except Exception as e:
    print(f"Error: {str(e)}")