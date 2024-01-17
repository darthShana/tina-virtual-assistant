from pydantic import BaseModel, Field
from langchain.agents import tool


@tool
def adjust_to_turners_geography(location: str) -> list[str]:
    """Fetch the turners branches for a given location.
    If the human is in a specific locations use this tool will find the relevant turners branches for that location,
    so it can be used in subsequent actions to find vehicles.
    """
    if location.casefold() == 'Auckland'.casefold():
        return ['Westgate', 'North Shore', 'Otahuhu', 'Penrose', 'Botany', 'Manukau']

    if location.casefold() == 'Wellington'.casefold():
        return ['Wellington', 'Porirua']

    return [location]
