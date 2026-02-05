from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = JsonOutputParser()

template = PromptTemplate(
    template="give me 5 facts about {topic}\n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
res = chain.invoke({'topic':'black hole'})

print(res)