from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load your extracted text
with open("agriculture_knowledge.txt", "r", encoding="utf-8") as f:
    data = f.read()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(data)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build and save vector DB
db = FAISS.from_texts(texts, embedding_model)
db.save_local("faiss_agriculture")

print("✅ Vector database saved!")
