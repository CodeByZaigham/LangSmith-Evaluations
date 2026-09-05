from langchain.agents import create_agent
from langchain_groq import ChatGroq
from tools import search_tool,get_weather
from dotenv import load_dotenv
load_dotenv()
import os

llm=ChatGroq(model="openai/gpt-oss-120b")

os.environ["LANGCHAIN_PROJECT"] = "Agentic Application"

agent = create_agent(
    model=llm,
    tools=[search_tool,get_weather],
    system_prompt="you are a helpful city weather and news assistant"
)

config={
    "run_name":"agentic trace"
}

print("welcome to the city agent, press 0 to exit")
while True:
    query=input("ask weather or news about a city: ")
    if query=="0": break
    result=agent.invoke({
        "messages":[{"role":"user","content":query}]
    },config=config)
    print(result["messages"][-1].content)