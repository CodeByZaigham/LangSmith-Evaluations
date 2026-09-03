from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage 
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

print("enter 1 for logical reasoning mode\n")
print("enter 2 for creative mode\n")
print("enter 3 for coding purposes\n")
print("enter 4 for funny chatbot\n")

choice = input("enter your choice: ")
mode = ""

if choice == "1":
    mode = "You are a logical reasoning expert. Answer every question step-by-step with clear logic."

elif choice == "2":
    mode = "You are a highly creative storyteller. Give imaginative, unique, and expressive answers."

elif choice == "3":
    mode = "You are an expert software engineer. Provide clean, optimized, and well-explained code."

elif choice == "4":
    mode = "You are a funny chatbot. Respond with humor, jokes, and a lighthearted tone."

else:
    mode = "Invalid choice. Please restart and select a valid option."

history=[
     SystemMessage(content=mode)
]

print("welcome to our chatbot! press 0 to exit application")
while True:
     prompt=input("you: ")
     if prompt=="0": break
     history.append(HumanMessage(content=prompt))
     response=model.invoke(history)
     history.append(AIMessage(content=response.content))
     print("bot: ",response.content)