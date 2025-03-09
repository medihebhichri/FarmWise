import http.client

conn = http.client.HTTPSConnection("yahoo-weather5.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "28cc3e82d1msh11305df1d0a25a5p14eb0ajsn0d469506dc13",
    'x-rapidapi-host': "yahoo-weather5.p.rapidapi.com"
}

conn.request("GET", "/weather?location=ariana&format=json&u=f", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))