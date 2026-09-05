from langsmith import traceable
from state import state
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()

llm=ChatGroq(model="openai/gpt-oss-120b")

class EvaluationSchema(BaseModel):
     feedback: str = Field(description="Detailed feedback for the essay")
     score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = llm.with_structured_output(EvaluationSchema)

@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state:state):
     prompt = (
          "Evaluate the language quality of the following essay and provide feedback "
          "and assign a score out of 10.\n\n" + state["essay"]
     )
     out = structured_model.invoke(prompt)
     return {"language_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state:state):
     prompt = (
          "Evaluate the depth of analysis of the following essay and provide feedback "
          "and assign a score out of 10.\n\n" + state["essay"]
     )
     out = structured_model.invoke(prompt)
     return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state:state):
     prompt = (
          "Evaluate the clarity of thought of the following essay and provide feedback "
          "and assign a score out of 10.\n\n" + state["essay"]
     )
     out = structured_model.invoke(prompt)
     return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}

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