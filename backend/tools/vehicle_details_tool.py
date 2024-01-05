from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import tool
from pydantic.v1 import BaseModel, Field
from tinydb import TinyDB, Query

db = TinyDB('tinydb/db.json')


class VehicleDetailsInput(BaseModel):
    source: str = Field(description="the source URL for the vehicle the query is about")
    query: str = Field(description="the query to answer about a specific vehilcle")


@tool(args_schema=VehicleDetailsInput)
def vehicle_details(source: str, query: str) -> str:
    """
    useful for when you need to answer queries about a specific vehicle including its contacts,
    features, condition and guarantees.
    """
    q = Query()
    vehicle_description = db.search(q.source == source)

    template = """
         given the following Vehicle details
         ===
         {vehicle_description}
         ===
         answer the query:{query}
     """

    prompt = PromptTemplate(
        template=template,
        input_variables=["vehicle_description", "query"],
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")
    chain = LLMChain(llm=llm, prompt=prompt)
    res = chain.run(query=query, vehicle_description=vehicle_description[0]['content'])
    return res


if __name__ == "__main__":
    print("in here product details")
    print(vehicle_details("https://www.turners.co.nz/Cars/Used-Cars-for-Sale/holden/astra/23856565", "who is the consultant"))
