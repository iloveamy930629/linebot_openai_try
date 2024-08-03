# from flask import Flask, request, abort

# from app import (
#     LineBotApi, WebhookHandler
# )
# from linebot.exceptions import (
#     InvalidSignatureError
# )
# from linebot.models import *

# #======python的函數庫==========
# import tempfile, os
# import datetime
# import openai
# import time
# import traceback

# from pymongo import MongoClient
# #======python的函數庫==========


# app = Flask(__name__)
# static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')
# # Channel Access Token
# line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
# # Channel Secret
# handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
# # OPENAI API Key初始化設定
# openai.api_key = os.getenv('OPENAI_API_KEY')

# # MongoDB Connection
# mongo_client = MongoClient(os.getenv('MONGODB_URI'))
# db = mongo_client.get_database('sample_restaurants')  # Replace 'your_database_name' with your database name
# collection = db.get_collection('restaurants')  # Replace 'your_collection_name' with your collection name

# # def GPT_response(text):
# #     # 接收回應
# #     response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=text, temperature=0.5, max_tokens=500)
# #     print(response)
# #     # 重組回應
# #     answer = response['choices'][0]['text'].replace('。','')
# #     return answer

# def GPT_response(text, additional_info=None):
#     # Combine the text with additional information
#     if additional_info: 
#         # 設定附加信息的最大長度
#         max_additional_info_length = 1000  # 例：限制的長度
#         if len(additional_info) > max_additional_info_length:
#             additional_info = additional_info[:max_additional_info_length] + "..."
        
#         prompt = f"{text}\n\nAdditional Info:\n{additional_info}"
#     else:
#         prompt = text

#     try:
#         # 從 OpenAI 生成回應
#         response = openai.Completion.create(
#             model="gpt-3.5-turbo-instruct",
#             prompt=prompt,
#             temperature=0.5,
#             max_tokens=500  # 根據需要調整 max_tokens
#         )
#         print(response)
#         # 重構回應
#         answer = response['choices'][0]['text'].strip()
#         return answer
#     except openai.error.OpenAIError as e:
#         print(f"OpenAI API 錯誤: {e}")
#         return "OpenAI API 錯誤: " + str(e)

# # 監聽所有來自 /callback 的 Post Request
# @app.route("/callback", methods=['POST'])
# def callback():
#     # get X-Line-Signature header value
#     signature = request.headers['X-Line-Signature']
#     # get request body as text
#     body = request.get_data(as_text=True)
#     app.logger.info("Request body: " + body)
#     # handle webhook body
#     try:
#         handler.handle(body, signature)
#     except InvalidSignatureError:
#         abort(400)
#     return 'OK'


# # 處理訊息
# @handler.add(MessageEvent, message=TextMessage)
# def handle_message(event):
#     msg = event.message.text.lower()
#     try:
#         # Search for related restaurants in MongoDB
#         query = {'cuisine': {'$regex': msg, '$options': 'i'}}  # Case-insensitive search for cuisine
#         restaurants = list(collection.find(query))
        
#         if restaurants:
#             # Format the restaurant information
#             additional_info_text = "\n".join([
#                 f"Restaurant: {r['name']}, Cuisine: {r['cuisine']}, Address: {r['address'].get('street', 'N/A')}, Borough: {r.get('borough', 'N/A')}"
#                 for r in restaurants
#             ])
#         else:
#             additional_info_text = 'No related restaurants found'

#         # Pass the additional info to the GPT_response function
#         GPT_answer = GPT_response(msg, additional_info_text)
#         print(GPT_answer)
#         line_bot_api.reply_message(event.reply_token, TextSendMessage(GPT_answer))
#     except:
#         print(traceback.format_exc())
#         line_bot_api.reply_message(event.reply_token, TextSendMessage('你所使用的OPENAI API key額度可能已經超過，請於後台Log內確認錯誤訊息'))
        

# @handler.add(PostbackEvent)
# def handle_message(event):
#     print(event.postback.data)


# @handler.add(MemberJoinedEvent)
# def welcome(event):
#     uid = event.joined.members[0].user_id
#     gid = event.source.group_id
#     profile = line_bot_api.get_group_member_profile(gid, uid)
#     name = profile.display_name
#     message = TextSendMessage(text=f'{name}歡迎加入')
#     line_bot_api.reply_message(event.reply_token, message)
        
        
# import os
# if __name__ == "__main__":
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)
