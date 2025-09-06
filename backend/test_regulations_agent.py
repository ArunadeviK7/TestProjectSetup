import os
import tempfile
from regulations_agent import RegulationsAssimilationAgent

def test_process_regulation():
    # Create a temporary regulation document
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as tmp:
        tmp.write("This regulation requires fairness and transparency. GDPR compliance is mandatory.")
        document_path = tmp.name
    # Initialize agent
    agent = RegulationsAssimilationAgent()
    # Call process_regulation
    result = agent.process_regulation(
        name="GDPR",
        region="EU",
        effective_from="2018-05-25",
        document_path=document_path
    )
    print("Test Result:", result)
    os.remove(document_path)

if __name__ == "__main__":
    test_process_regulation()
