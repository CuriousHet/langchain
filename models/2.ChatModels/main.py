from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# --- Section 1: Chat Model Initialization ---
# Chat Models are distinct from LLMs in that they handle "messages" (System, User, AI) rather than just raw text strings.
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.3)

# --- Section 2: Simple Invocation ---
print("--- Simple Question ---")
result = model.invoke('What is langraph?')
print(result.content)
print("-" * 50)

# --- Section 3: Advanced Structure with Messages ---
# Using SystemMessage allows us to set the behavior/persona of the AI.
print("\n--- Role-Playing Experiment ---")
messages = [
    SystemMessage(content="You are a sarcastic senior software engineer. Respond with dry humor."),
    HumanMessage(content="Why is my code confusing?")
]

# The model will verify the context (SystemMessage) and answer the HumanMessage
result_structured = model.invoke(messages)
print(f"User: Why is my code confusing?")
print(f"AI: {result_structured.content}")
