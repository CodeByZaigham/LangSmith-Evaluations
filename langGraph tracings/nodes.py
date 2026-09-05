from langsmith import traceable
from state import state
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()
import json

llm=ChatGroq(model="openai/gpt-oss-120b")

@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state:state):
     prompt = (
          "Evaluate the language quality of the following essay and provide feedback,"
          "and assign a score out of 10.\n\n" + state["essay"],
          "strictly give response in json format like: 'feedback': .. , 'score': .. "
     )
     response = llm.invoke(prompt)
     data=json.loads(response.content)
     return {"language_feedback": data["feedback"], "individual_scores": [int(data["score"])]}

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state:state):
     prompt = (
          "Evaluate the depth of analysis of the following essay and provide feedback "
          "and assign a score out of 10.\n\n" + state["essay"],
          "strictly give response in json format like: 'feedback': .. , 'score': .. "
     )
     response = llm.invoke(prompt)
     data=json.loads(response.content)
     return {"analysis_feedback":data["feedback"], "individual_scores": [int(data["score"])]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state:state):
     prompt = (
          "Evaluate the clarity of thought of the following essay and provide feedback "
          "and assign a score out of 10.\n\n" + state["essay"],
          "strictly give response in json format like: 'feedback': .. , 'score': .. "
     )
     response = llm.invoke(prompt)
     data=json.loads(response.content)
     return {"clarity_feedback": data["feedback"], "individual_scores": [int(data["score"])]}

@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state:state):
     prompt = (
          "Based on the following feedback, create a summarized overall feedback.\n\n"
          f"Language feedback: {state.get('language_feedback','')}\n"
          f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
          f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
     )
     overall = llm.invoke(prompt).content
     scores = state.get("individual_scores", []) or []
     avg = (sum(scores) / len(scores)) if scores else 0.0
     return {"overall_feedback": overall, "avg_score": avg}