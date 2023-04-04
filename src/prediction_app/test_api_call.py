import requests
import pdb
import json

# IMPORTANT: this script can only be executed if app.py has been executed before hand to deploy the app

test_input = {
    "id": 1001,
    "accommodates": 4,
    "room_type": "Entire home/apt",
    "beds": 2,
    "bedrooms": 1,
    "bathrooms": 2,
    "neighbourhood": "Brooklyn",
    "tv": 1,
    "elevator": 1,
    "internet": 0,
    "latitude": 40.71383,
    "longitude": -73.9658,
}

headers = {
    "content-type": "application/json",
    "cache-control": "no-cache",
}


if __name__ == "__main__":
    try:
        r = requests.get(
            url="http://127.0.0.1:3000/predictions",
            headers=headers,
            data=json.dumps(test_input),
        )
        print(r.json())
    except Exception as e:
        print(e)
