import typing
import logging
from langgraph.graph import StateGraph, END
from .state import EdgecaseNegativecaseState
from .node import read_requirements, generate_edgecase_negative_cases, push_to_testrail

# Logger setup (inherits from nodes.py but repeat here for clarity)
logging.basicConfig(level=logging.INFO, format="🔹 %(message)s")
logger = logging.getLogger(__name__)

def build_graph():
    #Create graph object with our state type
    workflow = StateGraph(EdgecaseNegativecaseState)

    #Register nodes
    workflow.add_node("read_requirements", read_requirements)
    workflow.add_node("generate_edgecase_negative_cases", generate_edgecase_negative_cases)
    workflow.add_node("push_to_testrail", push_to_testrail)

    #Define edges(basically flow of execution)
    workflow.set_entry_point("read_requirements")
    workflow.add_edge("read_requirements", "generate_edgecase_negative_cases")
    workflow.add_edge("generate_edgecase_negative_cases", "push_to_testrail")       
    workflow.add_edge("push_to_testrail", END)

    #Compile the graph to excutable app 
    app = workflow.compile()
    logger.info("✅ Test Case Generator pipeline built successfully")
    return app