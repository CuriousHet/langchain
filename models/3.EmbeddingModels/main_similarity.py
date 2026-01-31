from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# --- Configuration ---
# Google specific: You can specify 'task_type' for better optimization.
query_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", task_type="RETRIEVAL_QUERY"
)
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"
)

# --- Data Preparation ---
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Bumrah is a indian bowler"

# --- Embedding ---
# Convert documents and query to vectors
d_embed = doc_embeddings.embed_documents(documents)
q_embed = query_embeddings.embed_query(query)

# --- Similarity Calculation ---
print(f"Query: '{query}'\n")

scores = []
for i, d in enumerate(d_embed):
    # Calculate cosine similarity: ranges from -1 (opposite) to 1 (identical)
    similarity = cosine_similarity([q_embed], [d])[0][0]
    scores.append((documents[i], similarity))

# --- Sorting and Displaying Results ---
# Sort by similarity score in descending order (Best match first)
scores.sort(key=lambda x: x[1], reverse=True)

print("--- Ranked Results ---")
for doc, score in scores:
    print(f"Score: {score:.4f} | Document: {doc}")

# --- Experiments ---
# 1. Try changing the query to "Who is the captain cool?" (Should match Dhoni high).
# 2. Try a query unrelated to cricket, e.g., "Python programming", and see low scores.