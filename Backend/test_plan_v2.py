import requests
import json
import sys

try:
    print("Testing /triage/plan endpoint...")
    response = requests.get("http://127.0.0.1:8000/triage/plan")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        planning = data.get("planning", [])
        print(f"Planning items: {len(planning)}")
        if len(planning) > 0:
            print("First item:", json.dumps(planning[0], indent=2))
        else:
            print("Planning list is empty.")
    else:
        print("Error response:", response.text)
except Exception as e:
    print(f"Exception: {e}")
