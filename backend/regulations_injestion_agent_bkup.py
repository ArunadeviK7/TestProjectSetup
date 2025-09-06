import os
import glob
import re
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from py2neo import Graph, Node, Relationship
import openai
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")  # optional
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
SENTENCE_EMBEDDING_MODEL = os.getenv("SENTENCE_EMBEDDING_MODEL")
openai.api_type = "azure"
openai.api_base = "https://tcoeaiteam-41mini.openai.azure.com/"
openai.api_version = "2025-01-01-preview"
openai.api_key = "77nnrf6UeMb2ChlR7r8xLqzXerm9hbPpeinWan7C5s7LCN4jVHwoJQQJ99BFACYeBjFXJ3w3AAABACOGVWQY"
client = AzureOpenAI(
    azure_endpoint="https://tcoeaiteam-41mini.openai.azure.com/",
    api_key="77nnrf6UeMb2ChlR7r8xLqzXerm9hbPpeinWan7C5s7LCN4jVHwoJQQJ99BFACYeBjFXJ3w3AAABACOGVWQY",
    api_version="2025-01-01-preview",
)

deployment_id = "gpt-4.1-nano"  # Your Azure deployment name


# Load transformer model for semantic similarity

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

facet_descriptions = {
    "transparency": "clarity, explainability, and interpretability of AI decisions",
    "fairness": "equitable treatment, bias mitigation, and nondiscrimination",
    "privacy": "protection of personal data and user information",
    "accountability": "clear responsibility and oversight mechanisms",
    "safety": "protection from harm, reliability, and robustness",
}

facet_embeddings = {
    facet: sbert_model.encode(desc, convert_to_tensor=True)
    for facet, desc in facet_descriptions.items()
}

graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class RegulationsIngestionAgent:
    def __init__(self, graph, sbert_model, facet_embeddings, client, deployment_id):
        self.graph = graph
        self.sbert_model = sbert_model
        self.facet_embeddings = facet_embeddings
        self.client = client
        self.deployment_id = deployment_id

    def extract_responsible_ai_facet(self, text, threshold=0.4):
        embedding = self.sbert_model.encode(text, convert_to_tensor=True)
        matches = []
        for facet, f_embedding in self.facet_embeddings.items():
            sim = util.pytorch_cos_sim(embedding, f_embedding).item()
            if sim > threshold:
                matches.append(facet)
        return matches

    def parse_documents(self, folder_path):
        documents = {}
        for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                documents[os.path.basename(file_path)] = f.read()
        return documents

    def get_compliance_actions(self, article_text, facet_name):
        prompt = f"""You are a Responsible AI compliance auditor. The following is an AI regulation article that relates to the facet **{facet_name}**.\nBased on this article, identify what actions an AI team should take to ensure compliance.\nReturn the response in this structured format:\n- **[Action Category]**: [One-line description of the audit, assessment, or validation step]\nKeep the response concise, with 2–3 items only.\nArticle Content:\n\"\"\"{article_text}\"\"\"\n"""
        chat_prompt = [{"role": "system", "content": [
            {"type": "text", "text": prompt}]}]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id, messages=chat_prompt, temperature=0.3
            )
            text = response.choices[0].message.content.strip()
            lines = [line for line in text.splitlines()
                     if line.startswith("- **")]
            actions = []
            for line in lines:
                match = re.match(r"- \*\*(.+?)\*\*: (.+)", line)
                if match:
                    category, description = match.groups()
                    actions.append({"category": category.strip(),
                                   "description": description.strip()})
            return actions
        except Exception as e:
            return [{"category": "LLM Error", "description": str(e)}]

    def build_knowledge_graph(self, documents):
        print("🧹 Clearing existing graph...")
        self.graph.delete_all()
        node_counts = {"Regulation": 0, "Article": 0, "Facet": 0}
        relationship_counts = {"CONTAINS": 0, "RELATES_TO": 0}
        created_facets = set()
        print("🚧 Building new graph...")
        for doc_title, content in documents.items():
            print(f"\n📘 Processing Regulation: {doc_title}")
            regulation_node = Node("Regulation", name=doc_title)
            self.graph.create(regulation_node)
            node_counts["Regulation"] += 1
            articles = re.split(r"(Article \d+[:\.\s])", content)
            article_count = 0
            for i in range(1, len(articles), 2):
                article_title = articles[i].strip()
                article_body = articles[i + 1].strip()
                article_count += 1
                print(f"  ✏️  Creating Article Node: {article_title}")
                article_node = Node(
                    "Article", title=article_title, text=article_body)
                self.graph.create(article_node)
                self.graph.create(Relationship(
                    regulation_node, "CONTAINS", article_node))
                node_counts["Article"] += 1
                relationship_counts["CONTAINS"] += 1
                matched_facets = self.extract_responsible_ai_facet(
                    article_body)
                for facet in matched_facets:
                    if facet not in created_facets:
                        print(f"    🧩 Creating new Facet Node: {facet}")
                        facet_node = Node("Facet", name=facet)
                        self.graph.create(facet_node)
                        created_facets.add(facet)
                        node_counts["Facet"] += 1
                    else:
                        facet_node = self.graph.nodes.match(
                            "Facet", name=facet).first()
                    print(f"    🔗 Linking Article to Facet: {facet}")
                    self.graph.create(Relationship(
                        article_node, "RELATES_TO", facet_node))
                    relationship_counts["RELATES_TO"] += 1
                    actions = self.get_compliance_actions(article_body, facet)
                    for action in actions:
                        action_node = Node(
                            "ComplianceAction",
                            category=action["category"],
                            description=action["description"],
                            facet=facet,
                            article=article_title,
                        )
                        self.graph.create(action_node)
                        self.graph.create(Relationship(
                            article_node, "RECOMMENDS", action_node))
                        self.graph.create(Relationship(
                            action_node, "SUPPORTS", facet_node))
                        print(
                            f"    📋 {action['category']}: {action['description']}")
            print(f"✅ {article_count} Articles processed under {doc_title}")
        print("\n📊 Graph Build Complete. Summary:")
        for node_type, count in node_counts.items():
            print(f"  🔹 Nodes [{node_type}]: {count}")
        for rel_type, count in relationship_counts.items():
            print(f"  🔸 Relationships [{rel_type}]: {count}")

    def print_graph(self):
        print("In Graph print")
        node_count = self.graph.evaluate("MATCH (n) RETURN count(n)")
        rel_count = self.graph.evaluate("MATCH ()-[r]->() RETURN count(r)")
        print(f"\n✅ Graph built successfully:")
        print(f"  🔹 Total Nodes: {node_count}")
        print(f"  🔹 Total Relationships: {rel_count}")
        facets = ["transparency", "fairness",
                  "privacy", "accountability", "safety"]
        for facet in facets:
            query = f"""
            MATCH (a:Article)-[:RELATES_TO]->(f:Facet {{name: '{facet}'}})
            RETURN a.title AS Article, a.text AS Snippet
            """
            results = self.graph.run(query).data()
            if results:
                df = pd.DataFrame(results)
                df["Snippet"] = df["Snippet"].apply(
                    lambda x: x[:120].replace(
                        "\n", " ") + "..." if len(x) > 120 else x
                )
                print(f"\n--- Articles related to {facet.upper()} ---")
                print(df.to_string(index=False))
            else:
                print(f"No articles found for the facet: {facet}")


# LangGraph StateGraph workflow for regulations ingestion


def step_parse_documents(state):
    agent = state['agent']
    folder_path = state['folder_path']
    documents = agent.parse_documents(folder_path)
    state['documents'] = documents
    print(f"Parsed {len(documents)} documents.")
    return state


def step_build_knowledge_graph(state):
    agent = state['agent']
    documents = state['documents']
    agent.build_knowledge_graph(documents)
    print("Knowledge graph built.")
    return state


def step_print_graph(state):
    agent = state['agent']
    agent.print_graph()
    print("Graph printed.")
    return state


def regulations_ingestion_workflow(folder_path):
    agent = RegulationsIngestionAgent(
        graph, sbert_model, facet_embeddings, client, deployment_id)
    state = {
        'agent': agent,
        'folder_path': folder_path
    }
    workflow = StateGraph(dict)
    workflow.add_node('parse_documents', step_parse_documents)
    workflow.add_node('build_knowledge_graph', step_build_knowledge_graph)
    workflow.add_node('print_graph', step_print_graph)
    workflow.set_entry_point('parse_documents')
    workflow.add_edge('parse_documents', 'build_knowledge_graph')
    workflow.add_edge('build_knowledge_graph', 'print_graph')
    workflow.add_edge('print_graph', END)
    app = workflow.compile()
    final_state = app.invoke(state)
    return final_state['agent']


# Example usage
if __name__ == "__main__":
    # Step 1: Run agentic ingestion workflow with LangGraph StateGraph
    regulations_ingestion_workflow("./regulations")
    # Step 2: Map project requirements (use attempt_reg.py for this)
    # ...existing code...
