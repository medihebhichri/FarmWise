import requests
import json

API_KEY = "0028163a63814f85999222341250602"
LOCATION = "Tunisia kasserine"
URL = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={LOCATION}&aqi=yes"

try:
    response = requests.get(URL)
    response.raise_for_status()
    data = response.json()
    weather_info = {
        "location": {
            "name": data["location"]["name"],
            "region": data["location"]["region"],
            "country": data["location"]["country"],
            "lat": data["location"]["lat"],
            "lon": data["location"]["lon"],
            "tz_id": data["location"]["tz_id"],
            "localtime": data["location"]["localtime"],
        },
        "current": {
            "last_updated": data["current"]["last_updated"],
            "temp_c": data["current"]["temp_c"],
            "temp_f": data["current"]["temp_f"],
            "is_day": data["current"]["is_day"],
            "condition": {
                "text": data["current"]["condition"]["text"],
                "icon": data["current"]["condition"]["icon"],
            },
            "wind_kph": data["current"]["wind_kph"],
            "wind_mph": data["current"]["wind_mph"],
            "humidity": data["current"]["humidity"],
            "feelslike_c": data["current"]["feelslike_c"],
            "feelslike_f": data["current"]["feelslike_f"],
            "vis_km": data["current"]["vis_km"],
            "vis_miles": data["current"]["vis_miles"],
            "air_quality": data.get("current", {}).get("air_quality", {}),  # Handles cases where AQI might be missing
        }
    }
    print(json.dumps(weather_info, indent=4))

except requests.exceptions.RequestException as e:
    print(f"Error fetching weather data: {e}")
