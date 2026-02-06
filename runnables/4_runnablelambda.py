from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

def normalize_and_validate(input: dict) -> dict:
    topic = input.get("topic", "").strip()

    if not topic:
        raise ValueError("Topic must not be empty")

    return {
        "topic": topic.title(),
        "topic_length": len(topic),
        "original_input": input
    }

def choose_fact_count(data: dict) -> dict:
    if data["topic_length"] > 10:
        count = 1
    else:
        count = 3

    return {
        **data,
        "fact_count": count
    }

def post_process(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]

normalize_input = RunnableLambda(normalize_and_validate)
decide_fact_count = RunnableLambda(choose_fact_count)
format_output = RunnableLambda(post_process)


prompt = PromptTemplate(
    template="""
Create {fact_count} concise but informative facts about {topic}.
Focus on accuracy and real-world relevance.
""",
    input_variables=["topic", "fact_count"]
)

generation_chain = (
    prompt
    | model
    | parser
)

chain = (
    normalize_input
    | decide_fact_count
    | RunnableParallel({
        "metadata": RunnablePassthrough(),
        "facts": generation_chain | format_output
    })
)

result = chain.invoke({"topic": "artificial intelligence"})
print(result)
