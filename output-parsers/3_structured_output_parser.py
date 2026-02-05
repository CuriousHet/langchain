from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

schemas = [
    ResponseSchema(name="tool", description="Tool name"),
    ResponseSchema(name="purpose", description="What it is used for")
]

parser = StructuredOutputParser.from_response_schemas(schemas)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a documentation bot"),
    ("human", """
Explain Git.

{format_instructions}
""")
])

res = model.invoke(
    prompt.format_messages(
        format_instructions=parser.get_format_instructions()
    )
)

print(parser.parse(res.content))
