import os
import glob
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from py2neo import Graph, Node, Relationship
import openai
from openai import AzureOpenAI
openai.api_type = "azure"
openai.api_base = "https://tcoeaiteam-41mini.openai.azure.com/"
openai.api_version = "2025-01-01-preview"
openai.api_key = "77nnrf6UeMb2ChlR7r8xLqzXerm9hbPpeinWan7C5s7LCN4jVHwoJQQJ99BFACYeBjFXJ3w3AAABACOGVWQY"
client = AzureOpenAI(
    azure_endpoint="https://tcoeaiteam-41mini.openai.azure.com/",
    api_key="77nnrf6UeMb2ChlR7r8xLqzXerm9hbPpeinWan7C5s7LCN4jVHwoJQQJ99BFACYeBjFXJ3w3AAABACOGVWQY",
    api_version="2025-01-01-preview",
)

# openai.api_key = "dsfdsfd"

deployment_id = "gpt-4.1-nano"  # Your Azure deployment name


# Load transformer model for semantic similarity

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


# Define responsible AI facets with descriptions

facet_descriptions = {
    "transparency": "clarity, explainability, and interpretability of AI decisions",
    "fairness": "equitable treatment, bias mitigation, and nondiscrimination",
    "privacy": "protection of personal data and user information",
    "accountability": "clear responsibility and oversight mechanisms",
    "safety": "protection from harm, reliability, and robustness",
}

# Precompute facet embedding
facet_embeddings = {
    facet: sbert_model.encode(desc, convert_to_tensor=True)
    for facet, desc in facet_descriptions.items()
}

# Connect to Neo4j
graph = Graph("neo4j://127.0.0.1:7687", auth=("neo4j", "password"))
# Semantic facet extractor using SBERT


def extract_responsible_ai_facet(text, threshold=0.4):
    embedding = sbert_model.encode(text, convert_to_tensor=True)
    matches = []
    for facet, f_embedding in facet_embeddings.items():

        sim = util.pytorch_cos_sim(embedding, f_embedding).item()

        if sim > threshold:

            matches.append(facet)

    return matches
# Load regulation documents from folder


def parse_documents(folder_path):
    documents = {}
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            documents[os.path.basename(file_path)] = f.read()
    return documents


def get_compliance_actions(article_text, facet_name):
    prompt = f"""You are a Responsible AI compliance auditor. The following is an AI regulation article that relates to the facet **{facet_name}**.
    Based on this article, identify what actions an AI team should take to ensure compliance.
    Return the response in this structured format:
    - **[Action Category]**: [One-line description of the audit, assessment, or validation step]
    Keep the response concise, with 2–3 items only.
    Article Content:
    \"\"\"{article_text}\"\"\"
    """

    chat_prompt = [{"role": "system", "content": [
        {"type": "text", "text": prompt}]}]

    try:
        response = client.chat.completions.create(
            model=deployment_id, messages=chat_prompt, temperature=0.3
        )
        text = response.choices[0].message.content.strip()
        # Extract structured items from markdown-style bullets
        lines = [line for line in text.splitlines() if line.startswith("- **")]
        actions = []
        for line in lines:
            match = re.match(r"- \*\*(.+?)\*\*: (.+)", line)
            if match:
                category, description = match.groups()
                actions.append(
                    {"category": category.strip(), "description": description.strip()}
                )
        return actions
    except Exception as e:
        return [{"category": "LLM Error", "description": str(e)}]


def build_knowledge_graph(documents):
    print("🧹 Clearing existing graph...")
    graph.delete_all()
    node_counts = {"Regulation": 0, "Article": 0, "Facet": 0}
    relationship_counts = {"CONTAINS": 0, "RELATES_TO": 0}
    created_facets = set()
    print("🚧 Building new graph...")
    for doc_title, content in documents.items():
        print(f"\n📘 Processing Regulation: {doc_title}")
        regulation_node = Node("Regulation", name=doc_title)
        graph.create(regulation_node)
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
            graph.create(article_node)
            graph.create(Relationship(
                regulation_node, "CONTAINS", article_node))

            node_counts["Article"] += 1
            relationship_counts["CONTAINS"] += 1
            matched_facets = extract_responsible_ai_facet(article_body)
            for facet in matched_facets:
                if facet not in created_facets:
                    print(f"    🧩 Creating new Facet Node: {facet}")
                    facet_node = Node("Facet", name=facet)
                    graph.create(facet_node)
                    created_facets.add(facet)
                    node_counts["Facet"] += 1
                else:
                    facet_node = graph.nodes.match("Facet", name=facet).first()
                print(f"    🔗 Linking Article to Facet: {facet}")
                # Add as a relationship with compliance note
                # compliance_note = get_compliance_recommendation(article_body, facet)
                # Create relationship without compliance note
                graph.create(Relationship(
                    article_node, "RELATES_TO", facet_node))
                relationship_counts["RELATES_TO"] += 1

                # 🔍 Get LLM-based compliance actions
                actions = get_compliance_actions(article_body, facet)
                for action in actions:
                    action_node = Node(
                        "ComplianceAction",
                        category=action["category"],
                        description=action["description"],
                        facet=facet,
                        article=article_title,
                    )

                    graph.create(action_node)

                    # Connect Action to Article and Facet
                    graph.create(Relationship(
                        article_node, "RECOMMENDS", action_node))

                    graph.create(Relationship(
                        action_node, "SUPPORTS", facet_node))

                    print(
                        f"    📋 {action['category']}: {action['description']}")

                # rel = Relationship(article_node, "RELATES_TO", facet_node, compliance=compliance_note)
                # graph.create(rel)
                # relationship_counts["RELATES_TO"] += 1
                # print(f"    📋 Compliance Recommendation for {facet}:\n      {compliance_note.splitlines()[0]}...")

        print(f"✅ {article_count} Articles processed under {doc_title}")
    # Final Summary
    print("\n📊 Graph Build Complete. Summary:")
    for node_type, count in node_counts.items():
        print(f"  🔹 Nodes [{node_type}]: {count}")
    for rel_type, count in relationship_counts.items():
        print(f"  🔸 Relationships [{rel_type}]: {count}")


# Map requirements to RAI facets and matching guidelines
def map_project_requirements(requirements_text):

    print("\nMapping project requirements to Responsible AI facets and regulations:")
    matched_facets = extract_responsible_ai_facet(requirements_text)
    for facet in matched_facets:
        print(f"\nRequirement maps to facet: {facet}")
        facet_node = graph.nodes.match("Facet", name=facet).first()
        if facet_node:
            rels = graph.match((None, facet_node), r_type="RELATES_TO")
            seen_titles = set()
            for rel in rels:
                article_node = rel.start_node
                title = article_node.get("title", "Unknown Article")
                if title not in seen_titles:
                    snippet = article_node["text"][:200].replace("\n", " ")
                    print(f"  Suggested guideline from {title}: {snippet}...")
                    # Query for compliance actions recommended for this article
                    action_query = (
                        f"MATCH (a:Article {{title: '{title}'}})-[:RECOMMENDS]->(ca:ComplianceAction) "
                        f"RETURN ca.category AS Category, ca.description AS Description"
                    )
                    actions = graph.run(action_query).data()
                    if actions:
                        print(f"    Compliance Actions for {title}:")
                        for action in actions:
                            print(
                                f"      - [{action['Category']}] {action['Description']}")
                    else:
                        print(f"    No compliance actions found for {title}.")
                    seen_titles.add(title)

# Entry point for triggering build


def trigger_graph_build(folder_path="./regulations"):
    print(f"Scanning documents in: {folder_path}")
    documents = parse_documents(folder_path)
    print(f"Found {len(documents)} documents.")
    build_knowledge_graph(documents)
    print("Graph build complete.")


def print_graph():

    print("In Graph print")
    node_count = graph.evaluate("MATCH (n) RETURN count(n)")
    rel_count = graph.evaluate("MATCH ()-[r]->() RETURN count(r)")
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

        results = graph.run(query).data()
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


# Run directly
if __name__ == "__main__":

    # Step 1: Build graph
    # trigger_graph_build()
    # print_graph()
    # Step 2: Map project requirements
    sample_requirements = (
        "The system must ensure fairness in decisions and protect personal data."
    )
    map_project_requirements(sample_requirements)
