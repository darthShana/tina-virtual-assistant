from pydantic import BaseModel, Field
from langchain.agents import tool


@tool
def adjust_to_turners_geography(location: str) -> list[str]:
    """Fetch the turners branches for a given location.
    If a user asks to see vehicles at a specific locations use this tool to find the relevant turners branches in that location,
    so it can be used in subsequent tools to find vehicles
    """
    if location.casefold() == 'Auckland'.casefold():
        return ['Westgate', 'North Shore', 'Otahuhu', 'Penrose', 'Botany', 'Manukau']

    return [location]
