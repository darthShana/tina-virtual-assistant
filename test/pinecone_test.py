import os
import pinecone
from openai import OpenAI
from const import PRODUCT_INDEX_NAME


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


client = OpenAI()

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])

index = pinecone.Index(PRODUCT_INDEX_NAME)

query_response = index.query(
    vector=get_embedding("family car with 7 seats"),
    top_k=5,
    include_values=True,
    include_metadata=True,
    filter={
        "$or": [
            {"location": {"$in": ["Wellington", "Porirua"]}},
            {"seats": {"$eq": 7}}
        ]
        # "$or":(eq(\"location\", \"Wellington\"), eq(\"location\", \"Porirua\"))

    }
)

for match in query_response["matches"]:
    print(match["id"])
    print(match["score"])
    print(match["metadata"])
