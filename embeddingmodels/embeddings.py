from dotenv import load_dotenv

load_dotenv()
from langchain_groq import Groq

embeddings = OpenAIEmbeddings( 
    model = "text-embedding-3-large",
    dimensions=64
)

vector = embeddings.embed_query("you are going to school to study")

print(vector)