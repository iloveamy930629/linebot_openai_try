from openai import OpenAI
import pymongo
import pandas as pd
import warnings
import os
from pymongo import IndexModel
from dotenv import load_dotenv
load_dotenv()

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Replace with your actual API key
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

# Set your MongoDB connection string here
mongo_uri = "mongodb+srv://AmyChen:amy930629@linebotdb.ihw7ynt.mongodb.net/?retryWrites=true&w=majority&appName=lineBotDB"
mongo_client = get_mongo_client(mongo_uri)

if mongo_client:
    db = mongo_client['sample_restaurants']
    collection = db['restaurants']

    # Read data from MongoDB into a pandas DataFrame
    cursor = collection.find({}, {"_id": 0}).limit(40) # Adjust the query/filter as needed
    dataset_df = pd.DataFrame(list(cursor))

    print(f"Data loaded from MongoDB, number of records: {len(dataset_df)}")

    # Create embeddings for restaurant data (e.g., description or cuisine type)
    embeddings = []
    for index, row in dataset_df.iterrows():
        print(f"Processing row {index + 1}/{len(dataset_df)}: {row['name']}")
        embedding = get_embedding(row['cuisine'])
        embeddings.append(embedding)

    dataset_df['cuisine_embedding'] = embeddings
    print("Embedding generation completed")
    print(dataset_df.head())

    # Convert DataFrame to list of dictionaries
    documents = dataset_df.to_dict('records')

    # Insert documents into MongoDB
    collection.insert_many(documents)
    print("Data ingestion into MongoDB completed")




def vector_search(user_query, collection):
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
                # "cuisine_embedding": 1,
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

def handle_user_query(query, collection):
    search_results = vector_search(query, collection)

    result_str = ''
    for result in search_results:
        result_str += (
            f"Name: {result.get('name', 'N/A')}, "
            f"Address: {result.get('address', 'N/A')}, "
            f"Cuisine: {result.get('cuisine', 'N/A')}, "
            f"Borough: {result.get('borough', 'N/A')}\n"
        )

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a restaurant recommendation system."},
            {"role": "user", "content": "Answer this user query: " + query + " with the following context: " + result_str}
        ]
    )

    return completion.choices[0].message.content, result_str

query = "Where can I find a American restaurant if I am in Queens?"
response, source_information = handle_user_query(query, collection)

print(f"Response: {response}")
print(f"Source Information:\n{source_information}")
