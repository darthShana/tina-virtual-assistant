import requests
import os
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain_community.document_transformers import BeautifulSoupTransformer
from tinydb import TinyDB
import time

import pinecone

from const import PRODUCT_INDEX_NAME
from scraper.scrape_test import get_meta_schema, filter_content

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])


def run_crawler():
    db = TinyDB('tinydb/db.json')

    vgm_urls = [
        'https://www.turners.co.nz/Cars/Used-Cars-for-Sale/?sortorder=7&pagesize=40&pageno=1&issearchsimilar=true&types=wagon',
        'https://www.turners.co.nz/Cars/Used-Cars-for-Sale/?sortorder=7&pagesize=40&pageno=1&issearchsimilar=true&types=utility',
        'https://www.turners.co.nz/Cars/Used-Cars-for-Sale/?sortorder=7&pagesize=40&pageno=1&issearchsimilar=true&types=hatchback',
        'https://www.turners.co.nz/Cars/Used-Cars-for-Sale/?sortorder=7&pagesize=40&pageno=1&issearchsimilar=true&types=van',
        "https://www.turners.co.nz/Cars/Used-Cars-for-Sale/?sortorder=7&pagesize=40&pageno=1&issearchsimilar=true&types=sedan",
        "https://www.turners.co.nz/Cars/Used-Cars-for-Sale/?sortorder=7&pagesize=40&pageno=1&issearchsimilar=true&types=suv",
        "https://www.turners.co.nz/Cars/Used-Cars-for-Sale/?sortorder=7&pagesize=40&pageno=1&issearchsimilar=true&types=coupe"
    ]

    URLS = []

    for vgm_url in vgm_urls:
        html_text = requests.get(vgm_url).text
        soup = BeautifulSoup(html_text, 'html5lib')

        attrs = {
            'class': "green"
        }

        for listing in soup.find_all('a', attrs=attrs):
            URLS.append("https://www.turners.co.nz/"+listing['href'])

    print(f'loading {len(URLS)} vehicles')

    loader = AsyncHtmlLoader(URLS, default_parser="html5lib")
    data = loader.load()
    html2text = BeautifulSoupTransformer()
    docs_transformed = html2text.transform_documents(data)
    filter_content(docs_transformed)

    print(f"loaded {len(docs_transformed)} documents")

    for d in docs_transformed:
        db.insert({'source': d.metadata['source'], 'content': d.page_content})

    schema = get_meta_schema()
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
    enhanced_documents = document_transformer.transform_documents(docs_transformed)
    # chunks = [docs_transformed[x:x+10] for x in range(0, len(docs_transformed), 10)]
    # enhanced_documents = []
    # for chunk in chunks:
    #     enhanced_documents.extend(document_transformer.transform_documents(chunk))
    #     print(f"enhanced documents:{len(enhanced_documents)}")

    print("metadata tagged")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0,
                                                   separators=["\n\n", "\n", "  ", ""])
    documents = text_splitter.split_documents(documents=enhanced_documents)
    print(f"split into {len(documents)} chunks")

    print(f"inserting into pinecone {len(documents)} documents")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=PRODUCT_INDEX_NAME)
    print("done pushing to pinecone")


if __name__ == "__main__":
    print("hello turners langchain")
    run_crawler()
