import os

from langchain.chains import LLMChain, RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.agents import tool
from const import PRODUCT_INDEX_NAME


@tool
def knowledge_base_search(query: str):
    """
    useful for when you need find vehicles that match a prospects needs such as off-roading, fishing, racing, camping.
    DO NOT use this tool in the 'Needs analysis' stage. Send the input as a natural language query DO NOT send it json
    """
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=PRODUCT_INDEX_NAME, embedding=embeddings
    )

    prompt_template = """Filter the following pieces of context which are parts of vehicle descriptions based 
        on criteria specified in the question. If you don't know the answer, just say that you don't know, 
        don't try to make up an answer.
    
            {context}
            
            Question: {question}
            Answer with a numbered list where each item is a summary of the piece of context that matches, 
            including why the context matches.
            """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt}

    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4-1106-preview")

    knowledge_base = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(
            search_kwargs={'k': 5}
        ),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
        verbose=True
    )
    return knowledge_base({
        "query": query,
    })
