from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=100,  # Increased token limit for longer answers
    do_sample=True,      # Required for temperature to work
    temperature=0.7      # Balanced creativity
)

chat_model = ChatHuggingFace(llm=llm)

print("--- Querying Zephyr 7B ---")
result = chat_model.invoke("What is the capital of India and why is it famous?")
print(result.content)
print("-" * 50)
