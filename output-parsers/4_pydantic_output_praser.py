from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Job(BaseModel):
    role: str
    experience: int = Field(description="years of experience")
    skills: list[str] = Field(..., max_length=10)

parser = PydanticOutputParser(pydantic_object=Job)

