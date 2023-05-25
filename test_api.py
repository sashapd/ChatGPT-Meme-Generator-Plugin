import requests
import json

local = False

if local:
    url = 'http://localhost:8000/generate_meme'
else:
    # Set the URL for the meme generation API
    url = 'https://memepluginchatgpt.azurewebsites.net/generate_meme'

# Define the JSON body for the POST request
body = {
    'memeTemplateName': 'One Does Not Simply',
    'memeText': 'One does not simply\nTest an API',
    'memeUseCase': 'Demonstrate API usage'
}

# Send the POST request
response = requests.post(url, json=body)

# Check the response
if response.status_code == 200:
    print("API Call Success!")
    print("Response:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"API Call Failed. Status Code: {response.status_code}")
