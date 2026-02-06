from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Give 5 top research on {topic}",
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template='Explain these following research - {facts}',
    input_variables=['facts'],
)

fact_gen = RunnableSequence(prompt1, model, parser)
par_chain = ({
    'fact': RunnablePassthrough(),
    'explaination': RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(fact_gen, par_chain)


res = chain.invoke({'topic': 'quantum physics'})
print(res)