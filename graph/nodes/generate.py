from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState



def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]


    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"generation": generation, "question": question, "documents": documents}