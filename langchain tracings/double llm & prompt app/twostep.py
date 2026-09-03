from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["LANGCHAIN_PROJECT"] = "Two step application"

LLM1=ChatMistralAI(model_name="mistral-medium-latest")
LLM2=ChatGroq(model="openai/gpt-oss-120b")

parser=StrOutputParser()

prompt1=PromptTemplate(
     template="write a 300 words eassy on {topic}",
     input_variables=["topic"]
)

prompt2=PromptTemplate(
     template="summarize the given eassy: {eassy}",
     input_variables=["eassy"]
)

output=RunnableSequence(
     prompt1 | LLM1 | parser | prompt2 | LLM2 | parser
)

config={
     "run_name":"sequence chain"
}

response=output.invoke({"topic":"social media"},config=config)

print(response)