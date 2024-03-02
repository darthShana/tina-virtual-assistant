SONG_DATA_SOURCE = """\
```json
{{
    "content": "Lyrics of a song",
    "attributes": {{
        "artist": {{
            "type": "string",
            "description": "Name of the song artist"
        }},
        "length": {{
            "type": "integer",
            "description": "Length of the song in seconds"
        }},
        "genre": {{
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }}
    }}
}}
```\
"""

VEHICLE_DATA_SOURCE = """\
```json
{{
    "content": "Vehicle listings on a catalogue",
    "attributes": {{
    "make": {{
        "description": "the manufacture who made this item",
        "type": "string"
    }},
    "model": {{
        "description": "the model name of this item",
        "type": "string"
    }},
    "year": {{
        "description": "the year the item was made",
        "type": "integer"
    }},
    "fuel": {{
        "description": "the fuel type this vehicle uses. One of "Petrol", 'Diesel', 'Hybrid']",
        "type": "integer"
    }},
    "seats": {{
        "description": "the number of seats to carray passengers in this vehicle",
        "type": "integer"
    }},
    "odometer": {{
        "description": "the odometer reading showing the milage of this vehicle",
        "type": "integer"
    }},
    "price": {{
        "description": "the price this vehicle is on sale for",
        "type": "number"
    }},
    "location": {{
        "description": "the location this vehicle is at. One of ['Whangarei', 'Westgate', 'North Shore', 'Otahuhu', 'Penrose', 'Botany', 'Manukau', 'Hamilton', 'Tauranga', 'New Plymouth', 'Napier', 'Rotorua', 'Palmerston North', 'Wellington', 'Nelson', 'Christchurch', 'Timaru', 'Dunedin', 'Invercargill']",
        "type": "string"
    }},
    "vehicle_type": {{
        "description": "the type of vehicle this is. One of ['Wagon', 'Sedan' ,'Hatchback', 'Utility', 'Sports Car', 'Van', 'Tractor']",
        "type": "string"
    }},
    "colour": {{
        "description": "the color of the vehicle",
        "type": "string"
    }},
    "drive": {{
        "description": "the drive wheels on this vehicle One of ['Two Wheel Drive', 'Four Wheel Drive']",
        "type": "string"
    }}
}}
```\
"""

VEHICLE_ANSWER1 = """\
```json
{{
    "query": "recent with good fuel economy, bluetooth, reversing camera",
    "filter": "and(or(eq(\"vehicle_type\", \"Hatchback\"), eq(\"vehicle_type\", \"Sedan\")), in(\"location\", [\"Westgate\", \"North Shore\", \"Otahuhu\", \"Penrose\", \"Botany\", \"Manukau\"]))"
}}
```\
"""

VEHICLE_ANSWER2 = """\
```json
{{
    "query": "good safety rating and spacious boot",
    "filter": "and(
        eq(\"fuel\", \"Hybrid\"), 
        or(eq(\"vehicle_type\", \"Utility\"), eq(\"vehicle_type\", \"Wagon\")), 
        eq(\"location\", \"Christchurch\"), lt(\"price\", 15000))"
}}
```\
"""

NO_FILTER_ANSWER = """\
```json
{{
    "query": "a fun vehicle good for summer driving",
    "filter": "NO_FILTER"
}}
```\
"""

CAR_QUERY_EXAMPLES = [
    {
        "i": 1,
        "data_source": VEHICLE_DATA_SOURCE,
        "user_query": """
        a hybrid SUV or station wagon with good safety rating and spacious boot
         in any of these locations
         ["Christchurch"]
         for under 15k
        """,
        "structured_request": VEHICLE_ANSWER2,
    },
    {
        "i": 2,
        "data_source": VEHICLE_DATA_SOURCE,
        "user_query": "a fun vehicle good for summer driving",
        "structured_request": NO_FILTER_ANSWER,
    },
    {
        "i": 3,
        "data_source": VEHICLE_DATA_SOURCE,
        "user_query": """
            recent hatchback or sedan with good fuel economy, bluetooth, reversing camera
             in any of these locations
             ["Westgate", "North Shore", "Otahuhu", "Penrose", "Botany", "Manukau"]
        """,
        "structured_request": VEHICLE_ANSWER1,
    },
]