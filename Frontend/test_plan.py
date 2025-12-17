import requests
import json

try:
    response = requests.get("http://127.0.0.1:8000/triage/plan")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response Length: {len(data)}")
    if len(data) > 0:
        print("First item:", json.dumps(data[0], indent=2))
    else:
        print("Response is empty list")
except Exception as e:
    print(f"Error: {e}")
