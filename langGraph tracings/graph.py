import os
from edges import workflow

os.environ["LANGCHAIN_PROJECT"]="langGraph Parallel workflow"

print("\nWELCOME TO ESSAY EVALUATION SYSTEM!\n")
essay=input("write an essay to evaluate: ")

result = workflow.invoke(
     {"essay": essay},
     config={
          "run_name": "evaluate_essay",  # becomes root run name
          "tags": ["essay", "langgraph", "evaluation"],
          "metadata": {
               "essay_length": len(essay),
               "model": "openai/gpt-oss-120b",
               "dimensions": ["language", "analysis", "clarity"],
          },
     },
)

print("\n=== Evaluation Results ===")
print("Language feedback:\n", result.get("language_feedback", ""), "\n")
print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
print("Individual scores:", result.get("individual_scores", []))
print("Average score:", result.get("avg_score", 0.0))

#each graph execution is a trace
#a node is a run inside that trace
# you can also visualize the path taken