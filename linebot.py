from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
import openai
import pymongo
import pandas as pd
import warnings
import os
import traceback

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Line bot setup
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')

# MongoDB connection
mongo_uri = os.getenv('MONGODB_URI')
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client['sample_restaurants']
collection = db['restaurants']

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""
    if not text or not isinstance(text, str):
        return None
    try:
        embedding = openai.Embedding.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None

def vector_search(user_query):
    """Perform a vector search in the MongoDB collection based on the user query."""
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",  # Make sure this index is created on your MongoDB collection
                "queryVector": query_embedding,
                "path": "cuisine_embedding",
                "numCandidates": 150,
                "limit": 5
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": 1,
                "address": 1,
                "cuisine": 1,
                "borough": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    results = collection.aggregate(pipeline)
    return list(results)

def handle_user_query(query):
    search_results = vector_search(query)

    result_str = ''
    for result in search_results:
        result_str += (
            f"Name: {result.get('name', 'N/A')}, "
            f"Address: {result.get('address', 'N/A')}, "
            f"Cuisine: {result.get('cuisine', 'N/A')}, "
            f"Borough: {result.get('borough', 'N/A')}\n"
        )

    completion = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=f"Answer this user query: {query} with the following context: {result_str}",
        temperature=0.5,
        max_tokens=500
    )

    return completion.choices[0].text.strip(), result_str

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text.lower()
    try:
        response, source_information = handle_user_query(msg)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(response))
    except:
        print(traceback.format_exc())
        line_bot_api.reply_message(event.reply_token, TextSendMessage('An error occurred. Please try again later.'))

@handler.add(PostbackEvent)
def handle_postback(event):
    print(event.postback.data)

@handler.add(MemberJoinedEvent)
def welcome(event):
    uid = event.joined.members[0].user_id
    gid = event.source.group_id
    profile = line_bot_api.get_group_member_profile(gid, uid)
    name = profile.display_name
    message = TextSendMessage(text=f'{name} has joined the group.')
    line_bot_api.reply_message(event.reply_token, message)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
