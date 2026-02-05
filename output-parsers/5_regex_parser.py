from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import RegexParser

parser = RegexParser(
    regex=r"Score:\s*(\d+)",
    output_keys=["score"]
)

text = "Evaluation complete. Score: 87"
print(parser.parse(text))