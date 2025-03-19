import requests

def call_muse_prep_api():
    api_url = "https://theindexx--muse-prep-api-prepare-dev.modal.run"
    payload = {"input": "input_str"}
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status() # Raise an error for bad status codes
        return response.json() # Assuming the API returns a JSON response
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Example usage
result = call_muse_prep_api()
if result:
    print("API call successful. Response:")
    print(result)
else:
    print("API call failed.")
