import requests
import json
import time
import pandas as pd


def fetch_weather(api_url, params):
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def main():
    api_url = "http://api.weatherstack.com/current"
    cities = ["Tunis", "Ariana", "Ben Arous", "Manouba", "Bizerte", "Nabeul", "Zaghouan", "Beja", "Jendouba", "Kef",
              "Siliana", "Sousse", "Monastir", "Mahdia", "Sfax", "Kairouan", "Kasserine", "Sidi Bouzid", "Gafsa",
              "Tozeur", "Kebili", "Gabes", "Medenine", "Tataouine"]

    all_weather_data = []

    while True:
        dataset = []
        for city in cities:
            params = {
                "access_key": "469a87a1df0adcfbc8291ac15d4c6ee3",
                "query": city
            }
            weather_data = fetch_weather(api_url, params)

            if weather_data:
                # Print the raw response to inspect the structure
                print(json.dumps(weather_data, indent=4))  # Inspect the structure

                # Check if 'current' is in the response
                if "current" in weather_data:
                    city_weather = {
                        "City": city,
                        "Temperature": weather_data["current"]["temperature"],
                        "Weather Description": weather_data["current"]["weather_descriptions"][0],
                        "Humidity": weather_data["current"]["humidity"],
                        "Pressure": weather_data["current"]["pressure"]
                    }
                    dataset.append(city_weather)
                else:
                    print(f"Error: 'current' data not found for {city}")

        all_weather_data.extend(dataset)

        # Create a DataFrame
        df = pd.DataFrame(all_weather_data)

        # Save the DataFrame to an Excel file
        df.to_excel("weather_data.xlsx", index=False)

        print(f"Data saved to Excel for {len(dataset)} cities.")

        time.sleep(60)  # Wait for 1 minute before fetching data again


if __name__ == "__main__":
    main()
