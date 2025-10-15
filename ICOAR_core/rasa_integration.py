import requests
import json

# Rasa API URL (assuming Rasa is running locally on port 5005)
RASA_API_URL = "http://localhost:5005/webhooks/rest/webhook"

def send_message_to_rasa(message: str):
    """
    Sends a message to the Rasa API and returns the response.
    """
    payload = {
        "message": message,
    }
    
    try:
        response = requests.post(RASA_API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad responses
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def process_rasa_response(response):
    """
    Process the response from Rasa and return the text response.
    """
    if "error" in response:
        return f"Error: {response['error']}"
    
    # Assuming the response contains text from Rasa
    rasa_text = response[0].get('text', 'Sorry, no response from Rasa.')
    return rasa_text

def shadow_agent_task(user_input: str):
    """
    Simulate the user interacting with Rasa and analyze the feedback.
    """
    # Step 1: Send the user input to Rasa
    print(f"Shadow Agent sending message: {user_input}")
    rasa_response = send_message_to_rasa(user_input)
    
    # Step 2: Process Rasa's response
    response_text = process_rasa_response(rasa_response)
    
    # Step 3: Analyze the response for feedback (for example, detecting cyberbullying)
    feedback = analyze_response_for_feedback(response_text)
    
    # Step 4: Return the feedback (or further action)
    return feedback

def analyze_response_for_feedback(response_text):
    """
    Analyze the response for certain feedback, such as detecting sensitive topics (e.g., cyberbullying).
    """
    if "cyberbullying" in response_text.lower():
        return "Potentially a sensitive topic: Cyberbullying detected."
    
    return "Response is safe."


