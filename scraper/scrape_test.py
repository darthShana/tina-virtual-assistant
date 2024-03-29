import os
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.document_transformers.openai_functions import create_metadata_tagger

import pprint

from langchain_community.document_transformers import BeautifulSoupTransformer


def run_crawler():
    loader = AsyncHtmlLoader('https://www.turners.co.nz/Cars/Used-Cars-for-Sale/toyota/sienta/23650540', default_parser="html5lib")
    data = loader.load()
    html2text = BeautifulSoupTransformer()
    docs_transformed = html2text.transform_documents(
        data,
        tags_to_extract=["div", "section"],
        remove_lines=True,
    )

    filter_content(docs_transformed)

    print(f"loaded {len(docs_transformed)} documents")

    schema = get_meta_schema()
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
    enhanced_documents = document_transformer.transform_documents(docs_transformed)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=25,
                                                   separators=["\n\n", "\n", "  ", ""])
    documents = text_splitter.split_documents(documents=enhanced_documents)
    print(f"split into {len(documents)} chunks")

    for doc in documents:
        pprint.pprint(doc)
        print("------------")


def filter_content(docs_transformed):
    for d in docs_transformed:
        if "Start: Main Content Area" in d.page_content:
            d.page_content = d.page_content.split('Start: Main Content Area')[1]
        if "End: Main Content Area" in d.page_content:
            d.page_content = d.page_content.split('End: Main Content Area')[0]


def get_meta_schema():
    schema = {
        "properties": {
            "make": {
                "type": "string",
                "description": "the manufacture of the vehicle listed",
            },
            "model": {
                "type": "string",
                "description": "the model of the vehicle listed",
            },
            "year": {
                "type": "integer",
                "description": "the this vehicle was manufactures",
            },
            "fuel": {
                "type": "string",
                "description": "the fuel type this vehicle uses",
                "enum": ["Petrol", "Diesel", "Petrol Hybrid"]
            },
            "seats": {
                "type": "integer",
                "description": "the number of seats this vehicle has",
            },
            "odometer": {
                "type": "integer",
                "description": "odometer reading showing the vehicle milage",
            },
            "price": {
                "type": "number",
                "description": "the price this vehicle is on sale for",
            },
            "location": {
                "type": "string",
                "description": "the location this vehicle is ",
                "enum": ["Whangarei", "Westgate", "North Shore", "Otahuhu", "Penrose", "Botany", "Manukau", "Hamilton",
                         "Tauranga", "New Plymouth", "Napier", "Rotorua", "Palmerston North", "Wellington", "Porirua", "Nelson",
                         "Christchurch", "Timaru", "Dunedin", "Invercargill"]
            },
            "vehicle_type": {
                "type": "string",
                "description": "the type for vehicle this is. eg car, van or SUV",
                "enum": ["Wagon", "Sedan", "Hatchback", "Utility", "Sports Car", "Van", "Tractor"]
            },
            "colour": {
                "type": "string",
                "description": "the colour of this vehicle"
            },
            "drive": {
                "type": "string",
                "description": "describes the wheels that drive the vehicle",
                "enum": ["Two Wheel Drive", "Four Wheel Drive"]
            }

        },
        "required": ["make", "model", "location", "year"],
    }
    return schema


if __name__ == "__main__":
    print("hello turners langchain")
    run_crawler()
