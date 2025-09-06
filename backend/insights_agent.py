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

deployment_id = DEPLOYMENT_ID or "gpt-4.1-nano"


class InsightsAgent:
    def __init__(self, graph, sbert_model, client, deployment_id):
        self.graph = graph
        self.sbert_model = sbert_model
        self.client = client
        self.deployment_id = deployment_id

    def process(self, data):
        # Implement insights logic here
        return {"status": "success", "stage": "insights", "details": "Insights agent processed data."}


graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
insights_agent = InsightsAgent(graph, sbert_model, client, deployment_id)


def insights_workflow(data):
    return insights_agent.process(data)
