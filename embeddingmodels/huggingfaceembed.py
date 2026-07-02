from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings( 
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text = [
    "Hello how are you shreya?",
    "Why are you so much interested in coding!",
    "Are you all are very cute"
]


vector = embedding.embed_documents(text)

print(vector)