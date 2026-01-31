from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

# Invoke the model with a simple string prompt
print("--- Standard Output ---")
result = llm.invoke("what is langchain?")
print(result)
print("-" * 50)


# --- OPTION 2: Experimenting with Parameters (Temperature) ---
# Temperature controls the "creativity" or randomness of the model.
# 0.0 = Deterministic, focused, logical (good for factual queries).
# 1.0 = Creative, random, diverse (good for poems, stories).

# High Temperature (Creative)
print("\n--- High Temperature (Creative) Output ---")
llm_creative = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)
result_creative = llm_creative.invoke("Write a whimsical 2-line poem about a coding error.")
print(result_creative)

# Low Temperature (Focused)
print("\n--- Low Temperature (Focused) Output ---")
llm_focused = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
result_focused = llm_focused.invoke("What is 2 + 2? Answer with just the number.")
print(result_focused)