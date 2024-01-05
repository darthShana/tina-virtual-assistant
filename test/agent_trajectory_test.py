# import logging
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
import pytest
import json
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from backend.tina import Tina
from test.agent_cot_evaluator import StepNecessityEvaluator


@pytest.fixture
def sales_agent():
    tina = Tina()
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", output_key="output")
    # WARNING: this output parser is NOT reliable yet.
    # It makes assumptions about output from LLM which can break and throw an error

    return AgentExecutor(agent=tina.agent_chain, tools=tina.tools(), memory=memory, return_intermediate_steps=True, verbose=True)


def test_calculate_simple_trajectory_with_geography(sales_agent):
    result = sales_agent(
        {
            "input": "im looking for a economical toyota in auckland",
        }
    )

    evaluator = StepNecessityEvaluator()
    expected = [
        {
            "name": "adjust_to_turners_geography",
            "arguments": [{'location': 'auckland'}],
            "observation": ['Westgate', 'North Shore', 'Otahuhu', 'Penrose', 'Botany', 'Manukau']
        },
        {
            "name": "vehicle_search",
            "arguments": [{
                "query": "economical Toyota",
                "branches": ["Westgate", "North Shore", "Otahuhu", "Penrose", "Botany", "Manukau"]}
            ],
            "observation": """
            1. 2015 Toyota Corolla GX with 60,000km
            2. 2015 Toyota Corolla GX with 60,000km
            3. 2018 Toyota Prius C Hybrid with 35,000km
            """
        }
    ]

    evaluation_result = evaluator.evaluate_agent_trajectory(
        prediction=json.dumps(expected),
        input="im looking for a economical toyota in auckland",
        agent_trajectory=result['intermediate_steps'],
    )

    print(evaluation_result)
    assert evaluation_result['score'] == 1
