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
        {"input": "im looking for a red toyota in christchurch"},
        {"output": """
I found a couple of red Toyotas for you in Christchurch:

1. **2013 Toyota Prius Alpha**
   - Price: $14,400
   - Odometer: 97,310 km
   - Fuel: Petrol Hybrid
   - Seats: 5
   - Vehicle Type: Wagon
   - [View more details](https://www.turners.co.nz/Cars/Used-Cars-for-Sale/toyota/prius/23650474)

2. **2012 Toyota Aqua**
   - Price: $12,000
   - Odometer: 50,876 km
   - Fuel: Petrol Hybrid
   - Seats: 5
   - Vehicle Type: Hatchback
   - [View more details](https://www.turners.co.nz/Cars/Used-Cars-for-Sale/toyota/aqua/23632957)

Please note that the second option, the Toyota Avensis, is actually black, not red, so I've excluded it from the list. If you're interested in the red options, the Prius Alpha and Aqua are both available and looking sharp!
        """})

    result = sales_agent(
        {
            "input": "can i get a number to talk to someone about the first one?",
        }
    )

    evaluator = StepNecessityEvaluator()
    expected = [
        {
            "name": "vehicle_details",
            "arguments": [{
                "source": "https://www.turners.co.nz/Cars/Used-Cars-for-Sale/toyota/prius/23650474",
                "query": "contact number"
            }],
            "observation": """
            The contact number provided in the vehicle details is 0272739415. This number is associated with Daman Salwan,
            """
        }
    ]

    evaluation_result = evaluator.evaluate_agent_trajectory(
        prediction=json.dumps(expected),
        input="can i get a number to talk to someone about the first one?",
        agent_trajectory=result['intermediate_steps'],
    )

    print(evaluation_result)
    assert evaluation_result['score'] == 1

