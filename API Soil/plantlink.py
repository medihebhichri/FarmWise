import http.client
import json

conn = http.client.HTTPSConnection("plantlink.p.rapidapi.com")

# Properly formatted JSON payload
payload = json.dumps({
    "phone_number": "2231231234"
})

headers = {
    'x-rapidapi-key': "28cc3e82d1msh11305df1d0...",
    'x-rapidapi-host': "plantlink.p.rapidapi.com",
    'Content-Type': "application/json"
}

conn.request("POST", "/your-endpoint", body=payload, headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
