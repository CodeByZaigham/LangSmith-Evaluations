from typing import TypedDict,List,Annotated
from langgraph.graph import add_messages
import operator

class state(TypedDict):
     essay: str
     language_feedback: str
     analysis_feedback: str
     clarity_feedback: str
     overall_feedback: str
     individual_scores: Annotated[List[int], operator.add]
     avg_score: float