import logging 
from pprint import pprint
from src.graph.edgecase_negative_case_generator.graph import build_graph
from src.graph.edgecase_negative_case_generator.state import EdgecaseNegativecaseState

logging.basicConfig(level=logging.INFO, format="🔹 %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting Test Case Generator pipeline...")

    #build pipeline graph 
    app = build_graph()

    init_state = EdgecaseNegativecaseState = {}

    final_state = app.invoke(init_state)

     # Pretty print results for teaching clarity
    logger.info("✅ Pipeline finished. Final state below:")
    pprint(final_state)


if __name__ == "__main__":
    main()




