import json
import requests
import logging

import quart
import quart_cors
from quart import request
import pandas as pd
import pickle

logging.basicConfig(level=logging.INFO)  # or DEBUG, ERROR, WARNING, etc.
logger = logging.getLogger(__name__)

def generate_meme_link_from_id(meme_id, meme_text):
    # Define the URL
    url = "http://localhost:8080/templates/" + meme_id

    # Define the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Define the body of the POST request
    body = {
        "style": ["style1", "style2"],  # replace with actual styles
        "text": meme_text.split("\n"),  # replace with actual text lines
        "layout": "layout",  # replace with actual layout
        "font": "font",  # replace with actual font
        "extension": "extension",  # replace with actual extension
        "redirect": False
    }

    logger.info(body)

    # Send the POST request
    response = requests.post(url, headers=headers, data=json.dumps(body))

    # Check the response
    if response.status_code < 300:
        logger.info("Request was successful.")
        logger.info("Meme link: %s", response.json()['url'])
        return response.json()['url']
    else:
        logger.info("Request failed. Status code: %s", response.status_code)



print(generate_meme_link_from_id("saltbae", "salty"))