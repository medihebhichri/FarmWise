from httpx import Client
import json
import time

# Configure API endpoints
COUNTRIES_API = "https://countriesnow.space/api/v0.1/countries"
COORDINATES_API = "https://nominatim.openstreetmap.org/search"
SOIL_API = "https://api.openepi.io/soil/property"

# Configure target parameters
TARGET_COUNTRY = "Tunisia"
SOIL_PARAMS = {
    "depths": ["0-5cm", "100-200cm"],
    "properties": ["bdod", "phh2o"],
    "values": ["mean", "Q0.05"]
}


def validate_coordinates(lat: float, lon: float) -> bool:
    """Ensure coordinates are within Tunisia's approximate bounds"""
    return (30.0 < lat < 38.0) and (7.0 < lon < 12.0)


with Client(follow_redirects=True, timeout=30) as client:
    # Fetch Tunisian cities
    try:
        response = client.get(COUNTRIES_API)
        response.raise_for_status()
        countries_data = response.json().get("data", [])
        cities = next((c["cities"] for c in countries_data
                       if c.get("country", "").lower() == TARGET_COUNTRY.lower()), [])
    except Exception as e:
        print(f"Failed to fetch cities: {str(e)}")
        exit()

    if not cities:
        print("No cities found for Tunisia")
        exit()

    # Get city coordinates
    city_coordinates = []
    for city in cities:
        try:
            time.sleep(1)  # Respect Nominatim's rate limit
            response = client.get(
                COORDINATES_API,
                params={"q": f"{city}, {TARGET_COUNTRY}", "format": "json"},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            response.raise_for_status()

            if data := response.json():
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                if validate_coordinates(lat, lon):
                    city_coordinates.append({"name": city, "lat": lat, "lon": lon})
                else:
                    print(f"Invalid coordinates for {city}: {lat}, {lon}")
        except Exception as e:
            print(f"Skipping {city}: {str(e)}")
            continue

    # Query soil properties
    for city in city_coordinates:
        print(f"\nProcessing {city['name']} ({city['lat']:.4f}, {city['lon']:.4f})")

        try:
            response = client.get(
                SOIL_API,
                params={
                    "lat": city["lat"],
                    "lon": city["lon"],
                    **SOIL_PARAMS
                }
            )
            response.raise_for_status()
            soil_data = response.json()

            for layer in soil_data.get("properties", {}).get("layers", []):
                prop_name = layer.get("name", "Unknown")
                unit = layer.get("unit_measure", {}).get("mapped_units", "N/A")

                for depth in layer.get("depths", []):
                    depth_label = depth.get("label", "Unknown Depth")
                    for value_type in ["mean", "Q0.05"]:
                        if value := depth.get("values", {}).get(value_type):
                            print(f"  {prop_name} @ {depth_label}: {value_type} = {value} {unit}")

        except Exception as e:
            print(f"  Failed to retrieve soil data: {str(e)}")