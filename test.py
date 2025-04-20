from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Lire ton texte
with open("agriculture_knowledge.txt", "r", encoding="utf-8") as file:
    data = file.read()

# 2. Construire le prompt
template = """Voici un extrait de livre sur l'agriculture :

{text}

Question : {question}
Réponds de manière claire et concise.
"""
prompt = PromptTemplate(input_variables=["text", "question"], template=template)

# 3. Créer la chaîne LLM
llm = Ollama(model="mistral")
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Tester une question
question = "Qu'est-ce que l'agriculture durable ?"
response = chain.run({"text": data[:4000], "question": question})  # 4000 caractères max

print("\n🧠 Réponse:")
print(response)

