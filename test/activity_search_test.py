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


def test_remembers_conversation_history(memory, sales_agent):
    memory.save_context(
        {"input": "im looking for a vehicle which can tow my boat"},
        {"output": """
Sure thing! To help you find the right vehicle for towing your boat, I'll need a bit more information. Could you please tell me:

1. The approximate weight of your boat and trailer combined (so we can determine the towing capacity needed).
2. Your preferred vehicle type (e.g., SUV, truck).
3. Your location (to find vehicles nearby).
4. Any specific makes or models you're interested in.
5. Your budget range.

With these details, I can search for vehicles that match your towing needs and preferences.
        """})

    result = sales_agent(
        {
            "input": "im in hamilton, something with 4wd for around 15k",
        }
    )

    evaluator = StepNecessityEvaluator()
    expected = [
        {
            "name": "adjust_to_turners_geography",
            "arguments": [{'location': 'hamilton'}],
            "observation": ['hamilton']
        },
        {
            "name": "vehicle_search",
            "arguments": [{
                "query": "4WD vehicles with towing capacity suitable for a boat",
                "branches": ["hamilton"],
                "price": "around 15k"
            }],
            "observation": """
            1. 2002 Nissan Navara S/C C/S
            2. 2009 Audi Q7
            3. 2011 Volkswagen Amarok DC 4M 400 HL
            """
        }
    ]

    evaluation_result = evaluator.evaluate_agent_trajectory(
        prediction=json.dumps(expected),
        input="im in hamilton, something for around 25k",
        agent_trajectory=result['intermediate_steps'],
    )

    print(evaluation_result)
    assert evaluation_result['score'] == 1

