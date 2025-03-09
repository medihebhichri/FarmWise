import http.client

conn = http.client.HTTPSConnection("ambee-soil-data.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "28cc3e82d1msh11305df1d0a25a5p14eb0ajsn0d469506dc13",
    'x-rapidapi-host': "ambee-soil-data.p.rapidapi.com"
}

conn.request("GET", "/soil/latest/by-lat-lng?lng=78.96&lat=20.59", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))