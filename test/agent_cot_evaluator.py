import json
from typing import Any, Optional, Sequence, Tuple

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation import AgentTrajectoryEvaluator
from langchain.schema import AgentAction


class StepNecessityEvaluator(AgentTrajectoryEvaluator):
    """Evaluate the perplexity of a predicted string."""

    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.0)
        template = """Given the following two sequences of steps between '------', which are chains of steps of Actions taken to answer a customer query .

        DATA
        ------
        {expected_trajectory}
        ------
        {actual_trajectory}
        ------
        
        are steps equivalent? Where equivalence is based on the Action of the step its inputs they take. Provide the response as Verdict on a new line as a single "Y" for yes or "N" for no, as well as reasoning on a new line
        Verdict:
        Reasoning:
        
        Example:
        ------
        0: call Action=[TurnersGeography] with Input=[Auckland] returned observation = [The Turners locations in Auckland are at Penrose, North Shore, and Manukau.]
        1: call Action=[ProductSearchQuery] with Input=[economical Toyota in Penrose, North Shore or Manukau] returned observation = [Found several economical Toyota vehicles in Auckland. Listings include a 2015 Toyota Corolla, a 2017 Toyota Yaris, and a 2018 Toyota Prius.]
        ------
        0: call Action=[TurnersGeography] with Input=[Auckland] returned observation = [The Turners location for Auckland is Turners Cars Penrose, Turners Cars North Shore, and Turners Cars Manukau.]
        1: call Action=[ProductSearchQuery] with Input=[economical Toyota in Auckland] returned observation = [Found several Toyota models that are known for their fuel efficiency available in Auckland. Here are a few options: 1. 2015 Toyota Corolla GX, 2. 2017 Toyota Yaris, 3. 2018 Toyota Prius C.
        
        results
        Verdict: N
        Reasoning: While the actions taken are the same the in the two chains the input to the second step is different. While the first input is Auckland the second takes in specific locations in Auckland
        """
        self.chain = LLMChain.from_string(llm, template)

    def _evaluate_agent_trajectory(
            self,
            *,
            prediction: str,
            input: str,
            agent_trajectory: Sequence[Tuple[AgentAction, str]],
            reference: Optional[str] = None,
            **kwargs: Any,
    ) -> dict:
        expected_trajectory = json.loads(prediction)
        expected_vals = [
            f"{i}: call Action=[{step['name']}] with Input={step['arguments']} returned observation = [{step['observation']}]"
            for i, (step) in enumerate(expected_trajectory)
        ]
        actual_vals = [
            f"{i}: call Action=[{action.tool}] with Input=[{action.tool_input}] returned observation = [{observation}]"
            for i, (action, observation) in enumerate(agent_trajectory)
        ]

        expected_trajectory = "\n".join(expected_vals)
        actual_trajectory = "\n".join(actual_vals)

        print(f"expected: \n {expected_trajectory}")
        print(f"actual: \n {actual_trajectory}")

        response = self.chain.run(dict(expected_trajectory=expected_trajectory, actual_trajectory=actual_trajectory), **kwargs)
        decision = response.split("\n")[0].strip()
        score = 1 if decision == "Verdict: Y" else 0
        return {"score": score, "value": decision, "reasoning": response}


