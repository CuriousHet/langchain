from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 

# Initialize Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

query_text = "Until death, all defeat is psychological"

# --- Embed Query ---
# embed_query is used for a single string, typically the user's search input.
# It creates a vector representation of the query to match against document vectors.
result = embeddings.embed_query(query_text)

print(f"Query: '{query_text}'")
print(f"Vector Length: {len(result)}")
print(f"Vector Sample: {result[:5]}...")