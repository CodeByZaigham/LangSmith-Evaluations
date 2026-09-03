from document_loader import load_file
from embeddings import create_embeddings
from retriever import retrieve_embeddings
from LLM import get_LLM
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

path="RAG_appication/Ian Goodfellow, Yoshua Bengio, Aaron Courville - Deep Learning (2017, MIT).pdf"

os.environ["langsmith-tracings"]="RAG APPLICATION"

@traceable(name="RAG_Pipeline")
def run_pipeline(query:str):
    documents=load_file(path)
    db=create_embeddings(documents)
    chunks=retrieve_embeddings(query,db)
    return chunks

def format_docs(docs):
    return "".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    """You are a book question-answering assistant.

    You MUST answer the user's question using ONLY the provided retrieved passages from the book.

    STRICT RULES:
    - The retrieved passages are your ONLY source of information.
    - Never use prior knowledge or outside information.
    - Never fill missing information with assumptions.
    - Never hallucinate an answer.
    - If the answer is not explicitly supported by the passages, respond exactly:
    "I couldn't find the answer in the provided book content."
    - If only part of the answer is supported, answer only that part.
    - You may combine information from multiple retrieved passages if they collectively answer the question.
    - Do not claim that something is in the book unless it is supported by the retrieved passages.
    - Do not invent chapter names, page numbers, quotations, or citations.
    - Keep the answer clear and concise.

    RETRIEVED BOOK PASSAGES:
    {context}
    """
    ),
    ("human", "{question}")
])



# retriever_chain=RunnableParallel({
#     "context": ,
#     "question":RunnablePassthrough()
# })

parser=StrOutputParser()

print("WELCOME TO RAG CHATBOT, PRESS 0 TO EXIT")

chain= prompt | get_LLM() | parser

while True:
    query=input("ask any question related to the uploaded document: ")
    if query=="0": break
    context=format_docs(run_pipeline(query))
    config={"run_name":"rag chat"}
    response=chain.invoke({
        "question":query,
        "context":context
    },config=config)
    print(response)

