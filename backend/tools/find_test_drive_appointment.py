from typing import Optional
import json
from langchain.agents import tool
from pydantic.v1 import BaseModel, Field

import requests

API_ENDPOINT = "https://marsapi.uat.turners.co.nz/api/TurnersWeb/Appointment/BookedPeriods"


class TestDriveAppointmentSearchInput(BaseModel):
    branch_id: int = Field(description="the id of the branch in which the user is looking for a test drive appointment")
    good_number: int = Field(description="the good number of vehicle to be test driven")


@tool(args_schema=TestDriveAppointmentSearchInput)
def find_test_drive_appointment(branch_id: int, good_number: int) -> str:
    """Finds available test drive appointments for a given vehicle at a given branch"""

    return test_drive_appointments(branch_id, good_number)


def test_drive_appointments(branch_id, good_number):
    params = {
        'branchId': '98bc28bd-94b1-e011-bc2b-00155db05300',
        'fromlcl': '2024-02-19',
        'tolcl': '2024-02-26',
        'type': '2',
    }

    with open('tinydb/test_drive.json') as f:
        appointments = json.load(f)
        print(appointments)



if __name__ == '__main__':
    print(test_drive_appointments(2042,24480738))