import http.client

conn = http.client.HTTPSConnection("wft-geo-db.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "28cc3e82d1msh11305df1d0a25a5p14eb0ajsn0d469506dc13",
    'x-rapidapi-host': "wft-geo-db.p.rapidapi.com"
}

conn.request("GET", "/v1/geo/countries/TN/places", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))