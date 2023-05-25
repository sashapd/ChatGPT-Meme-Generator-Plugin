import aiohttp
import json
import requests
import logging

import quart
import quart_cors
from quart import request
import pandas as pd
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO)  # or DEBUG, ERROR, WARNING, etc.
logger = logging.getLogger(__name__)

from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

logger.info("MEME PLUGIN STARTED")

# constants
EMBEDDING_MODEL = "text-embedding-ada-002"
dataset_path = "data/output.tsv"
df = pd.read_csv(dataset_path, delimiter='\t')

# set path to embedding cache
embedding_cache_path = "data/memes_embeddings_cache.pkl"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(string, model = EMBEDDING_MODEL, embedding_cache=embedding_cache):
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        logger.warning("Started get_embedding!")
        embedding = get_embedding(string, model)
        logger.warning("Ending get_embedding!")
        embedding_cache[(string, model)] = embedding 
        #with open(embedding_cache_path, "wb") as embedding_cache_file:
        #    pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


names = df["name"].tolist()
examples = df["example"].tolist()

# get embeddings for all examples
example_embeddings = [embedding_from_string(example, EMBEDDING_MODEL, embedding_cache) for example in examples]

# get embeddings for all names
name_embeddings = [embedding_from_string(str(name), EMBEDDING_MODEL) for name in names]
    
# combine the embeddings with a 2:1 weight for name vs. example
combined_embeddings_both = [2*name_emb + example_emb for name_emb, example_emb in zip(name_embeddings, example_embeddings)]

def get_meme_from_strings(query_name, query_example, model=EMBEDDING_MODEL):
    """logger.info out the k nearest neighbors of a given string."""

    logger.info("Getting query example embedding")

    # get the embedding of the source example
    query_example_embedding = embedding_from_string(query_example, model=model)

    logger.info("Finished getting query example embedding")

    if query_name:  # non-empty string
        logger.info("Getting query name embedding")
        # get the embedding of the source name
        query_name_embedding = embedding_from_string(query_name, model=model)

        # combine the embeddings with a 2:1 weight for name vs. example
        combined_embeddings = combined_embeddings_both
        combined_query_embedding = 2*query_name_embedding + query_example_embedding
    else:  # if the query_name is an empty string, we only compute embeddings for "example"
        combined_embeddings = example_embeddings
        combined_query_embedding = query_example_embedding

    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(combined_query_embedding, combined_embeddings, distance_metric="cosine")

    # get indices of nearest neighbors (function from embeddings_utils.py)
    indx = np.argmin(distances)

    # logger.info out source string
    logger.info(f"Source string: {query_name} {query_example}")

    return indx

def preprocess_query(query):
    return '\n'.join([line for line in query.split("\n") if line != ''])

def get_meme_id(query_example, query_name="", query_use_case=""):
    index = get_meme_from_strings(
        query_name=query_name,  
        query_example=query_example,
    )
    return df.loc[index, 'id']

async def generate_meme_link_from_id(meme_id, meme_text):
    # Define the URL
    url = "https://api.memegen.link/templates/" + meme_id

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
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(body)) as response:
            # Check the response
            if response.status < 300:
                logger.info("Request was successful.")
                json_response = await response.json()
                logger.info("Meme link: %s", json_response['url'])
                return json_response['url']
            else:
                logger.info("Request failed. Status code: %s", response.status)


app = quart_cors.cors(quart.Quart(__name__), allow_origin="*")

@app.route("/generate_meme", methods=['POST'])
async def generate_meme():
    request = await quart.request.get_json(force=True)
    logger.info("===")
    memeText = request["memeText"]
    logger.info(memeText)
    if "memeTemplateName" in request:
        memeTemplateName = request["memeTemplateName"]
    else:
        memeTemplateName = ""
    logger.info("Meme Template: ")
    logger.info(memeTemplateName)
    logger.info("Getting meme id")
    memeText = preprocess_query(memeText)
    meme_id = get_meme_id(memeText, memeTemplateName)
    logger.info("Generating link")
    link = await generate_meme_link_from_id(meme_id, memeText)
    logger.info("Returning response")
    if link is None:
        logger.info("Response: fail")
        return quart.Response(response='BAD', status=400)
    logger.info("Response: success")
    return quart.jsonify({"meme_link": link}), 200

@app.get("/logo.png")
async def plugin_logo():
    filename = 'logo.png'
    return await quart.send_file(filename, mimetype='image/png')

@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    host = request.headers['Host']
    with open("./.well-known/ai-plugin.json") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/json")

@app.get("/openapi.yaml")
async def openapi_spec():
    host = request.headers['Host']
    with open("openapi.yaml") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/yaml")

@app.route('/legal')
async def serve_disclaimer():
    return await quart.send_from_directory(directory='.', file_name='legal_info.html')

def main():
    app.run(debug=True, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
