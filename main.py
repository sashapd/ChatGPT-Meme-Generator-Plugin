import json
import requests

import quart
import quart_cors
from quart import request
import pandas as pd
import pickle

from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

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
def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print("CALCULATING EMBEDDING!!!!!")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

def get_meme_from_strings(
    names: list[str],
    examples: list[str],
    query_name: str,
    query_example: str,
    k_nearest_neighbors: int = 1,
    model=EMBEDDING_MODEL,
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""

    # get embeddings for all examples
    example_embeddings = [embedding_from_string(example, model=model) for example in examples]

    # get the embedding of the source example
    query_example_embedding = embedding_from_string(query_example, model=model)

    if query_name:  # non-empty string
        # get embeddings for all names
        name_embeddings = [embedding_from_string(str(name), model=model) for name in names]
    
        # get the embedding of the source name
        query_name_embedding = embedding_from_string(query_name, model=model)

        # combine the embeddings with a 2:1 weight for name vs. example
        combined_embeddings = [2*name_emb + example_emb for name_emb, example_emb in zip(name_embeddings, example_embeddings)]
        combined_query_embedding = 2*query_name_embedding + query_example_embedding
    else:  # if the query_name is an empty string, we only compute embeddings for "example"
        combined_embeddings = example_embeddings
        combined_query_embedding = query_example_embedding

    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(combined_query_embedding, combined_embeddings, distance_metric="cosine")

    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # print out source string
    print(f"Source string: {query_name} {query_example}")

    # print out its k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_name == names[i] and query_example == examples[i]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # print out the similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        Name: {names[i]}
        Example: {examples[i]}
        Distance: {distances[i]:0.3f}"""
        )

    return indices_of_nearest_neighbors

def get_meme_id(query_example, query_name="", query_use_case=""):
    names = df["name"].tolist()
    examples = df["example"].tolist()
    index = get_meme_from_strings(
        names=names, 
        examples=examples,
        query_name=query_name,  
        query_example=query_example,
        k_nearest_neighbors=1,
    )[0]
    return df.loc[index, 'id']

def generate_meme_link_from_id(meme_id, meme_text):
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

    print(body)

    # Send the POST request
    response = requests.post(url, headers=headers, data=json.dumps(body))

    # Check the response
    if response.status_code < 300:
        print("Request was successful.")
        print("Meme link: ", response.json()['url'])
        return response.json()['url']
    else:
        print("Request failed. Status code: ", response.status_code)



app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")

@app.post("/generate_meme")
async def generate_meme():
    request = await quart.request.get_json(force=True)
    print("===")
    print(request["memeText"])
    print(request["memeTemplateName"])
    if "memeUseCase" in request.keys():
        print(request["memeUseCase"])
    meme_id = get_meme_id(request["memeText"], request["memeTemplateName"])
    link = generate_meme_link_from_id(meme_id, request["memeText"])
    if link is None:
        return quart.Response(response='BAD', status=400)
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

def main():
    app.run(debug=True, host="0.0.0.0", port=5003)

if __name__ == "__main__":
    main()
