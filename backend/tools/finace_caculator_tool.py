from typing import Optional

from langchain.agents import tool
from pydantic.v1 import BaseModel, Field

import requests

API_ENDPOINT = "https://www.turners.co.nz/Client/Finance/CalculateSubmit"


class FinanceCalculatorInput(BaseModel):
    price: int = Field(description="the price of the vehicle used to get finance options for a vehicle")
    term: int = Field(description="the period over which the finance is for in months")
    deposit: Optional[int] = Field(description="the deposit the customer will use as a deposit")


@tool(args_schema=FinanceCalculatorInput)
def finance_calculation(price: int, term: int, deposit: Optional[int] = 0) -> str:
    """Gets finance options for a vehicle with a purchase price over specified period"""

    params = {
        'blockId': '21911',
        'loanAmount': price,
        'depositAmount': deposit,
        'loanTermMonths': term,
        'DivisionId': '1000',
        'pageId': '61590'
    }
    return requests.post(API_ENDPOINT, params=params).json()
