import pytest
import json
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from backend.tina import Tina
from test.agent_cot_evaluator import StepNecessityEvaluator


@pytest.fixture
def memory():
    return ConversationBufferMemory(return_messages=True, memory_key="chat_history", output_key="output")


@pytest.fixture
def sales_agent(memory):
    tina = Tina()
    return AgentExecutor(agent=tina.agent_chain, tools=tina.tools(), memory=memory, return_intermediate_steps=True, verbose=True)


def test_multiple_criteria_flow(memory, sales_agent):
    memory.save_context(
        {"input": "i want a new car"},
        {"output": """Fantastic! I'd be happy to help you find a new car. Could you provide me with some details about what you're looking for? For example, are you interested in a specific make or model, do you have a preferred fuel type, any particular features you want, and what's your budget? The more details you give me, the better I can assist you in finding the perfect car."""}
    )
    memory.save_context(
        {"input": "somehting recent with good fuel economy"},
        {"output": """Got it! You're looking for a recent model with good fuel economy. To narrow down the options, could you let me know the following:

        1. What's your budget for the new car?
        2. Do you have a preference for a particular body type (e.g., sedan, SUV, hatchback)?
        3. Are you looking for a specific make or are you open to suggestions?
        4. Do you have a preferred year range for the "recent" model?
        5. Any specific features you must have (e.g., Bluetooth, navigation, backup camera)?
        
        This information will help me find vehicles that match your criteria.
        """}
    )
    memory.save_context(
        {"input": "im in auckland. i'd like a hatchback or a sedan, with airbags and a reversing camera"},
        {"output": """Great, we have several branches in Auckland where we can look for a hatchback or sedan with 
        airbags and a reversing camera. Before I proceed with the search, could you please let me know your 
        budget range and if you have a preference for the year of the vehicle? This will help me find the best options for you.
        """}
    )

    result = sales_agent(
        {
            "input": "under 20k",
        }
    )
    print(result)

    # evaluator = StepNecessityEvaluator()
    # expected = [
    #     {
    #         "name": "adjust_to_turners_geography",
    #         "arguments": [{'location': 'hamilton'}],
    #         "observation": ['hamilton']
    #     },
    #     {
    #         "name": "vehicle_search",
    #         "arguments": [{
    #             "query": "4WD vehicles with towing capacity suitable for a boat",
    #             "branches": ["hamilton"],
    #             "price": "around 15k"
    #         }],
    #         "observation": """
    #         1. 2002 Nissan Navara S/C C/S
    #         2. 2009 Audi Q7
    #         3. 2011 Volkswagen Amarok DC 4M 400 HL
    #         """
    #     }
    # ]
    #
    # evaluation_result = evaluator.evaluate_agent_trajectory(
    #     prediction=json.dumps(expected),
    #     input="im in hamilton, something for around 25k",
    #     agent_trajectory=result['intermediate_steps'],
    # )
    #
    # print(evaluation_result)
    # assert evaluation_result['score'] == 1

