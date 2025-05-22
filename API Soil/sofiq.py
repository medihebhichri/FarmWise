import http.client

conn = http.client.HTTPSConnection("sofiq.p.rapidapi.com")

payload = """{
    "https://www.bakingbusiness.com/ext/resources/2023/02/08/bunge_AdSt_oticki_LEAD.jpeg?height=667&t=1718203734&width=1080":
    "https://s3.sofiq.com.br/typebot/public/public/workspaces/clqdt0ig70001qm2qtdc7z9nl/typebots/clvdr7cqi00023tid7aquuyfs/results/a1ud9h7a2yxkkd0xomgm4kh8/408972685441309",
    "model": "insects",
    "number": "0000000000000",
    "name": "API CALL"
}"""

headers = {
    'x-rapidapi-key': "28cc3e82d1msh11305df1d...",
    'x-rapidapi-host': "sofiq.p.rapidapi.com",
    'Content-Type': "application/json"
}

conn.request("POST", "/", body=payload, headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
