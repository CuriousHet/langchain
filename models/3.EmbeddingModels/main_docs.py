from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 'models/gemini-embedding-001' maps text to a high-dimensional vector space.
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

documents = [
    "Today is Monday",
    "Today is Tuesday",
    "Today is April Fools day",
]

# --- Embed Documents ---
vectors = embeddings.embed_documents(documents)

# --- Analysis ---
print(f"Number of documents embedded: {len(vectors)}")
print(f"Dimension of each vector: {len(vectors[0])}")

print("\n--- Vector Sample (First 5 numbers of Document 1) ---")
print(vectors[0][:5]) 
