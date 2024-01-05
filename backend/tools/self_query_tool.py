import os
from typing import Optional

from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
import pinecone
from const import PRODUCT_INDEX_NAME


def search(query: str):
    embeddings = OpenAIEmbeddings()

    vectordb = Pinecone.from_existing_index(
        index_name=PRODUCT_INDEX_NAME, embedding=embeddings
    )

    metadata_field_info = [
        AttributeInfo(
            name="make",
            description="the manufacture who made this item",
            type="string"
        ),
        AttributeInfo(
            name="model",
            description="the model name of this item",
            type="string"
        ),
        AttributeInfo(
            name="year",
            description="the year the item was made",
            type="integer"
        ),
        AttributeInfo(
            name="fuel",
            description="the fuel type this vehicle uses. One of ['', '', '']",
            type="integer"
        ),
        AttributeInfo(
            name="seats",
            description="the number of seats to carray passengers in this vehicle",
            type="integer"
        ),
        AttributeInfo(
            name="odometer",
            description="the odometer reading showing the milage of this vehicle",
            type="integer"
        ),
        AttributeInfo(
            name="price",
            description="the price this vehicle is on sale for",
            type="number",
        ),
        AttributeInfo(
            name="location",
            description="the location this vehicle is at. One of ['Whangarei', 'Westgate', 'North Shore', 'Otahuhu', 'Penrose', 'Botany', 'Manukau', 'Hamilton', 'Tauranga', 'New Plymouth', 'Napier', 'Rotorua', 'Palmerston North', 'Wellington', 'Nelson', 'Christchurch', 'Timaru', 'Dunedin', 'Invercargill']",
            type="string"
        ),
        AttributeInfo(
            name="vehicle_type",
            description="the type of vehicle this is. One of ['Wagon', 'Sedan' ,'Hatchback', 'Utility', 'Van', 'Tractor']",
            type="string"
        ),
        AttributeInfo(
            name="colour",
            description="the color of the vehicle",
            type="string"
        ),
        AttributeInfo(
            name="drive",
            description="the drive wheels on this vehicle One of ['Two Wheel Drive', 'Four Wheel Drive']",
            type="string"
        )
    ]
    document_content_description = "Vehicle listings on a catalogue"
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        verbose=True
    )

    docs = retriever.get_relevant_documents(query)
    return docs


pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])


class VehicleSearchInput(BaseModel):
    """input schema for a function to which does vehicle search"""
    query: str = Field(description="""
        query to perform a vehicle search on as natual language
        """)
    branches: list[str] = Field(description="turners branches to filter the query results on")
    price: Optional[str] = Field(description="the price restrictions the of the user. For example under 25k or around 17k")


@tool(args_schema=VehicleSearchInput)
def vehicle_search(query: str, branches: list[str], price: Optional[str] = None):
    """
    useful for when you need find vehicles that match vehicle criteria such as make, model, year, fuel-type, location, price, mileage, seats.
    This tool and process a query sent as natural language query DO NOT send it json
    """
    if branches:
        query += f"""
         in any of these locations
         {",".join(branches)}
         ------
        """
    if price:
        query += " for "+price
    return search(query=query)


