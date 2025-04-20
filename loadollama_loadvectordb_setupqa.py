import requests
import spacy
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# --- API Keys and Endpoints ---
TREFLE_API_KEY = 'SqzHwV1cv-tJrazgTTr0JhLDBBbGxfWKtAy-0S1bg_Q'
TREFLE_API_URL = "https://trefle.io/api/v1/plants/search"
PERENUAL_API_KEY = "sk-MO0b68045a9b46d249904"
PERENUAL_API_URL = "https://perenual.com/api/species-list"

# --- Vector DB and Embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_agriculture", embedding_model, allow_dangerous_deserialization=True)

# --- LLM Setup ---
llm = Ollama(model="mistral")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

# --- Named Entity Extraction ---
def extract_plant_name(question):
    doc = nlp(question)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE", "NORP", "PERSON"]:
            return ent.text
    return None

# --- Trefle API Call ---
def get_trefle_info(plant_name):
    params = {'q': plant_name, 'token': TREFLE_API_KEY}
    response = requests.get(TREFLE_API_URL, params=params)
    if response.status_code == 200 and response.json()['data']:
        plant = response.json()['data'][0]
        return {
            'source': 'Trefle',
            'name': plant.get('common_name', 'N/A'),
            'description': plant.get('description', 'No description'),
            'watering': plant.get('watering', 'No watering info'),
            'soil': plant.get('soil', 'No soil info'),
            'climate': plant.get('climate', 'No climate info')
        }
    return None

# --- Perenual API Call ---
def get_perenual_info(plant_name):
    url = f"{PERENUAL_API_URL}?key={PERENUAL_API_KEY}&q={plant_name}"
    response = requests.get(url)
    if response.status_code == 200 and response.json().get("data"):
        plant = response.json()["data"][0]
        return {
            'source': 'Perenual',
            'name': plant.get('common_name', 'N/A'),
            'description': plant.get('description', 'No description'),
            'watering': plant.get('watering', 'No info'),
            'sunlight': plant.get('sunlight', 'No info'),
            'cycle': plant.get('cycle', 'N/A'),
            'care_level': plant.get('care_level', 'N/A')
        }
    return None

# --- Final RAG Logic ---
def ask_question(question: str):
    plant_name = extract_plant_name(question)

    api_chunks = []

    # Retrieve from APIs if plant name found
    if plant_name:
        trefle_data = get_trefle_info(plant_name)
        perenual_data = get_perenual_info(plant_name)
        if trefle_data:
            api_chunks.append(str(trefle_data))
        if perenual_data:
            api_chunks.append(str(perenual_data))

    # Retrieve from local vector DB
    faiss_result = qa_chain.invoke({"query": question})
    faiss_answer = faiss_result['result']

    # Combine API info + local info
    context = "\n".join(api_chunks) + "\n" + faiss_answer

    # Final prompt for LLM to generate best answer
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""You are an agriculture expert. Use the following context to answer the question accurately.
        Context:
        {context}

        Question: {question}
        Answer:"""
    )
    final_prompt = prompt_template.format(question=question, context=context)
    return llm(final_prompt)
