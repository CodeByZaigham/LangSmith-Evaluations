from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv

def get_LLM():
    return ChatMistralAI(model_name="mistral-medium-latest")