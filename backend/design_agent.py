# FastAPI endpoint removed; only agent logic remains

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from py2neo import Graph
import openai
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY
client = AzureOpenAI(
    azure_endpoint=openai.api_base,
    api_key=openai.api_key,
    api_version=openai.api_version,
)

deployment_id = DEPLOYMENT_ID


# --- Sub-agents ---
def regulations_impact_agent(data):
    # Dummy implementation: should analyze requirements and return impacted regulations
    requirements = data.get('planning_answers', [])
    # TODO: Replace with real logic
    return ["GDPR", "EU AI Act"] if requirements else []


def responsible_ai_facets_impact_agent(data):
    # Dummy implementation: should analyze requirements and return responsible AI facets
    requirements = data.get('planning_answers', [])
    # TODO: Replace with real logic
    return ["Fairness", "Transparency"] if requirements else []


def augmentations_required_agent(data):
    # Dummy implementation: should propose augmentations
    requirements = data.get('planning_answers', [])
    # TODO: Replace with real logic
    return ["Synthetic data", "Data balancing"] if requirements else []


def metrics_applicability_agent(data):
    # Dummy implementation: should propose metrics
    requirements = data.get('planning_answers', [])
    # TODO: Replace with real logic
    return ["Accuracy", "Fairness", "Robustness"] if requirements else []


class DesignAgent:
    def __init__(self, graph, sbert_model, client, deployment_id):
        self.graph = graph
        self.sbert_model = sbert_model
        self.client = client
        self.deployment_id = deployment_id

    def process(self, data):
        print('[DesignAgent] Received data:', data)
        print('[DesignAgent] Running augmentations_required_agent...')
        augmentations = augmentations_required_agent(data)
        print('[DesignAgent] Augmentations:', augmentations)
        print('[DesignAgent] Running metrics_applicability_agent...')
        metrics = metrics_applicability_agent(data)
        print('[DesignAgent] Metrics:', metrics)
        print('[DesignAgent] Running regulations_impact_agent...')
        regulations_impacted = regulations_impact_agent(data)
        print('[DesignAgent] Regulations impacted:', regulations_impacted)
        print('[DesignAgent] Running responsible_ai_facets_impact_agent...')
        responsible_ai_facets = responsible_ai_facets_impact_agent(data)
        print('[DesignAgent] Responsible AI facets:', responsible_ai_facets)
        result = {
            "status": "success",
            "stage": "design",
            "augmentations": augmentations,
            "metrics": metrics,
            "regulations_impacted": regulations_impacted,
            "responsible_ai_facets": responsible_ai_facets
        }
        print('[DesignAgent] Returning result:', result)
        return result


graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
design_agent = DesignAgent(graph, sbert_model, client, deployment_id)


def design_workflow(data):
    return design_agent.process(data)
