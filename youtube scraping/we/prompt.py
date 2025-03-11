from openai import OpenAI

# Set up your OpenAI API key
client = OpenAI(api_key="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")

# Now you can use the OpenAI API as needed
response = client.chat.completions.create(
    model="",
    messages=[
        {"role": "system",
         "content": "You are an expert in plant care and agriculture. Provide detailed and accurate plant care recommendations based on the given inputs."}, ],
    max_tokens=5
)

print(response.choices[0].text.strip())

#
