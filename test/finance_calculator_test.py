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


def test_calling_finance_calculator(memory, sales_agent):
    memory.save_context(
        {"input": "im looking for a red toyota in christchurch"},
        {"output": """
I found this red Toyota for you in Christchurch:

1. **2013 Toyota Prius Alpha**
   - Price: $14,400
   - Odometer: 97,310 km
   - Fuel: Petrol Hybrid
   - Seats: 5
   - Vehicle Type: Wagon
   - [View more details](https://www.turners.co.nz/Cars/Used-Cars-for-Sale/toyota/prius/23650474)
        """})

    result = sales_agent(
        {
            "input": "what would my repayments be if i were to finance over 3 years?",
        }
    )

    evaluator = StepNecessityEvaluator()
    expected = [
        {
            "name": "finance_calculation",
            "arguments": [{
                "price": 14400,
                "term": 36
            }],
            "observation": {
                'WeeklyPaymentAmount': '$115.68',
                'MonthlyPaymentAmount': '$503.21',
                'FinanceInterestRate': '12.95% Per Annum',
                'TotalInterestAmount': '$3,138.67',
                'RepaymentAmount': '$18,115.67',
                'TotalYears': '3 year term'
            }
        }
    ]

    evaluation_result = evaluator.evaluate_agent_trajectory(
        prediction=json.dumps(expected),
        input="what would my repayments be if i were to finance over 3 years?",
        agent_trajectory=result['intermediate_steps'],
    )

    print(evaluation_result)
    assert evaluation_result['score'] == 1

