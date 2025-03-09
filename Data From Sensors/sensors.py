import serial
import serial.tools.list_ports
import pandas as pd
import time
import os


def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "USB" in port.description or "UART" in port.description:
            return port.device
    return None



csv_filename = "weather_data.csv"
columns = ["Timestamp", "Temperature", "Humidity", "Pressure", "Rain Sensor", "Light Sensor", "Wind Speed",
           "Wind Bearing"]


if not os.path.exists(csv_filename):
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_filename, index=False)


esp_port = find_esp32_port()
if esp_port is None:
    print("ESP32 not found! Connect the device and try again.")
    exit()


ser = serial.Serial(esp_port, 115200, timeout=2)
print(f"Connected to ESP32 on {esp_port}")



def read_serial_data():
    while True:
        try:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue


            if "Temp:" in line:
                print(line)
                parts = line.replace("°C", "").replace("%", "").replace("hPa", "").split(", ")
                temp = float(parts[0].split(": ")[1])
                humidity = float(parts[1].split(": ")[1])
                pressure = float(parts[2].split(": ")[1])

                line2 = ser.readline().decode("utf-8").strip()
                rain_sensor = int(line2.split(": ")[1].split(",")[0])
                light_sensor = int(line2.split(": ")[2])

                line3 = ser.readline().decode("utf-8").strip()
                wind_speed = float(line3.split(": ")[1].split(" ")[0])

                line4 = ser.readline().decode("utf-8").strip()
                wind_bearing = float(line4.split(": ")[1].split(" ")[0])


                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


                new_data = pd.DataFrame(
                    [[timestamp, temp, humidity, pressure, rain_sensor, light_sensor, wind_speed, wind_bearing]],
                    columns=columns)


                new_data.to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"Saved: {new_data.values[0]}")

        except Exception as e:
            print(f"Error: {e}")


# Start reading
read_serial_data()
