from backend.tools.self_query_tool import vehicle_search
from langchain.evaluation import load_evaluator


def test_self_query():
    result = vehicle_search({
        "query": "hatchback or sedan with airbags and a reversing camera",
        "branches": ['Westgate', 'North Shore', 'Otahuhu', 'Penrose', 'Botany', 'Manukau'],
        "price": "under 20k"
    })

    evaluator = load_evaluator("labeled_criteria", criteria="correctness")

    # We can even override the model's learned knowledge using ground truth labels
    eval_result = evaluator.evaluate_strings(
        input="hatchback or sedan with airbags and a reversing camera",
        prediction=result,
        reference="There should a vehicle available which match the query criteria",
    )
    print(f'With ground truth: {eval_result}')
    assert eval_result['score'] == 1


