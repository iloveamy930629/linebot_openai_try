from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
#======python的函數庫==========
import tempfile, os
import datetime
# import openai
import time
import traceback

import pymongo
from openai import OpenAI
from pymongo import MongoClient
import pandas as pd
import warnings

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')
# Line bot setup
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# OpenAI API Key
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# MongoDB connection
mongo_uri = os.getenv('MONGODB_URI')
mongo_client = pymongo.MongoClient(mongo_uri)
# db = mongo_client['sample_restaurants']
# collection = db['restaurants']
db = mongo_client['NTU_data']
collection = db['NTU_website_data']
# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""
    if not text or not isinstance(text, str):
        return None
    try:
        embedding = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None

def vector_search(user_query, collection):
    """Perform a vector search in the MongoDB collection based on the user query."""
    query_embedding = get_embedding(user_query)
    print(f"Query embedding: {query_embedding}")
    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",  # Make sure this index is created on your MongoDB collection
                "queryVector": query_embedding,
                "path": "website_data",
                "numCandidates": 30,
                "limit": 3
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": 1,
                "description": 1,
                "link": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    try:
        results = collection.aggregate(pipeline)
        result_list = list(results)
        print(f"Results: {result_list}")
        return result_list
    except Exception as e:
        print(f"Error in vector_search: {e}")
        return []
    
def remove_duplicate_urls(results):
    seen_urls = set()
    unique_results = []
    for result in results:
        url = result.get('link')
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    return unique_results

def handle_user_query(query, collection):
    search_results = vector_search(query, collection)
    # unique_results = remove_duplicate_urls(search_results)
    if not search_results:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一個台大學生專屬的客服機器人，請試著幫使用者解決問題並給予關心溫暖"},
                {"role": "user", "content": query}
            ]
        )
        return completion.choices[0].message.content, ''
    
    result_str = ''
    print(f"Search results: {search_results}")
    for result in search_results:
        result_str += (
            f"Name: {result.get('name', 'N/A')}, "
            f"Description: {result.get('description', 'N/A')}, "
            f"Link: {result.get('link', 'N/A')}\n"
        )
    detailed_response = f"用戶查詢：{query}\n\n資料庫相關資料：\n{result_str}\n請妥善利用資料庫資料提供一個簡潔且相關的回應來回答用戶的查詢，儘量幫助用戶解決問題，並適時提供資料庫中的所需相關鏈接。"

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一個台大學生專屬的客服機器人，請試著幫使用者解決問題並給予關心溫暖"},
            {"role": "user", "content": detailed_response}
        ]
    )

    return completion.choices[0].message.content, result_str

def summarize_text(text):
    summary_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"總結以下文本的關鍵要點，並生成出通順精要的回應文具：\n\n{text}，記得保留必要的網站連接，但不要列出太多冗長的聯絡資訊例如地址電話"}]
    )

    return summary_completion.choices[0].message.content


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
    collection = db['NTU_website_data']
    try:
        response, source_information = handle_user_query(msg, collection)
        print(f"Response: {response}")
        print(f"Source Information:\n{source_information}")
        summarized_response = summarize_text(response)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(summarized_response))
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
    message = TextSendMessage(text=f'{name}歡迎加入')
    line_bot_api.reply_message(event.reply_token, message)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
