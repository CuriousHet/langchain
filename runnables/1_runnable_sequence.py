from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableSequence
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

class StockDecision(BaseModel):
    analysis: str = Field(
        description="Concise stock analysis under 100 words"
    )
    decision: Literal["Buy", "Sell", "Hold", "CantSay"] = Field(
        description="Final trading decision"
    )
parser = PydanticOutputParser(pydantic_object=StockDecision)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

decision_prompt = PromptTemplate(
    template="""
You are a stock analyst.

Provide:
1. A concise stock analysis (max 100 words)
2. A final trading decision

{format_instructions}

Stock: {stock}
""",
    input_variables=["stock"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = RunnableSequence(decision_prompt, model, lambda x: x.content, parser)

res = chain.invoke({"stock":"BEL"})
print(res)

# res =chain.batch([{"stock": "BEL"}, {"stock": "TCS"}])
# print(res)