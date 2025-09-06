

import os
import glob
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
import openai
from sentence_transformers import SentenceTransformer, util
from py2neo import Graph, Node, Relationship
from langsmith import traceable, LangSmithTracer
import pandas as pd

load_dotenv()

class RegulationsAssimilationAgent:
    tracer = LangSmithTracer()
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.facet_embeddings = {
            facet: self.sbert_model.encode(desc, convert_to_tensor=True)
            for facet, desc in self.facet_descriptions.items()
        }
        openai.api_key = self.openai_api_key

    def create_supervisor(self):
        # Create LangGraph agentic supervisor with OpenAI LLM
        openai.api_key = self.openai_api_key
        supervisor = AgenticSupervisor(llm="openai", api_key=self.openai_api_key)
        return supervisor

    @traceable(tracer)
    def agentic_workflow(self, document_path: str) -> Dict[str, Any]:
        # Define agentic workflow steps
        workflow = AgenticWorkflow([
            self.parse_document,
            self.extract_responsible_ai_facets,
            self.determine_compliance_actions,
            self.build_knowledge_graph
        ])
        result = workflow.run(document_path)
        return result

    def parse_document(self, document_path: str) -> str:
        # Use OpenAI LLM to parse document and extract text
        with open(document_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Extract the main points and regulatory requirements from this document."},
                      {"role": "user", "content": content}]
        )
        return response.choices[0].message["content"]

    def extract_responsible_ai_facets(self, parsed_text: str) -> Dict[str, Any]:
        # Use LLM to extract Responsible AI facets
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "List Responsible AI facets mentioned in this text."},
                      {"role": "user", "content": parsed_text}]
        )
        facets = response.choices[0].message["content"]
        return {"facets": facets}

    def determine_compliance_actions(self, parsed_text: str) -> Dict[str, Any]:
        # Use LLM to determine compliance actions
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Suggest compliance actions based on this text."},
                      {"role": "user", "content": parsed_text}]
        )
        actions = response.choices[0].message["content"]
        return {"compliance_actions": actions}

    def build_knowledge_graph(self, parsed_text: str) -> Dict[str, Any]:
        # Use Neo4j to build knowledge graph
        with self.neo4j_driver.session() as session:
            # Example: create nodes and relationships
            session.run("CREATE (r:Regulation {text: $text})", text=parsed_text)
            # You can extend this to extract entities and relationships from parsed_text
        return {"knowledge_graph": "Neo4j graph updated"}

    def process_regulation(self, name: str, region: str, effective_from: str, document_path: str) -> Dict[str, Any]:
        # Add regulation node to Neo4j
        with self.neo4j_driver.session() as session:
            session.run("CREATE (r:Regulation {name: $name, region: $region, effectiveFrom: $effective_from, document: $document_path})",
                        name=name, region=region, effective_from=effective_from, document_path=document_path)
        # Run agentic workflow
        workflow_result = self.agentic_workflow(document_path)
        return workflow_result
    def parse_documents(self, folder_path: str) -> Dict[str, str]:
        documents = {}
        for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                documents[os.path.basename(file_path)] = f.read()
        return documents

    def extract_responsible_ai_facet(self, text: str, threshold: float = 0.4) -> List[str]:
        embedding = self.sbert_model.encode(text, convert_to_tensor=True)
        matches = []
        for facet, f_embedding in self.facet_embeddings.items():
            sim = util.pytorch_cos_sim(embedding, f_embedding).item()
            if sim > threshold:
                matches.append(facet)
        return matches

    def get_compliance_actions(self, article_text: str, facet_name: str) -> List[Dict[str, str]]:
        prompt = f"""You are a Responsible AI compliance auditor.
            The following is an AI regulation article that relates to the facet **{facet_name}**.
            Based on this article, identify what actions an AI team should take to ensure compliance.
            Return the response in this structured format:- **[Action Category]**: [One-line description of the audit, assessment, or validation step]
            Keep the response concise, with 2–3 items only.
            Article Content: \"\"\"{article_text}\"\"\"
            """
        chat_prompt = [
            {
                "role": "system",
                "content": prompt
            }
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=chat_prompt,
                temperature=0.3
            )
            # Extract structured items from markdown-style bullets
            text = response.choices[0].message.content.strip()
            lines = [line for line in text.splitlines() if line.startswith("- **")]
            actions = []
            for line in lines:
                match = re.match(r"- \*\*(.+?)\*\*: (.+)", line)
                if match:
                    category, description = match.groups()
                    actions.append({"category": category.strip(), "description": description.strip()})
            return actions
        except Exception as e:
            return [{"category": "LLM Error", "description": str(e)}]
    def build_knowledge_graph(documents):
        print("🧹 Clearing existing graph...")

        self.graph.delete_all()
        node_counts = {"Regulation": 0, "Article": 0, "Facet": 0}
        relationship_counts = {"CONTAINS": 0, "RELATES_TO": 0}
        created_facets = set()
        print("🚧 Building new graph...")

        for doc_title, content in documents.items():
            regulation_node = Node("Regulation", name=doc_title)
            self.graph.create(regulation_node)
            node_counts["Regulation"] += 1
            for i in range(1, len(articles), 2):
                article_title = articles[i].strip()
                article_body = articles[i + 1].strip()
                article_node = Node("Article", title=article_title, text=article_body)
        )
        chat_prompt = [
            {
                "role": "system",
                "content": prompt
            }
        ]
                self.graph.create(article_node)
                self.graph.create(Relationship(regulation_node, "CONTAINS", article_node))
                node_counts["Article"] += 1
                relationship_counts["CONTAINS"] += 1
                matched_facets = self.extract_responsible_ai_facet(article_body)
                for facet in matched_facets:
                    if facet not in created_facets:
                        facet_node = Node("Facet", name=facet)
                        self.graph.create(facet_node)
                        created_facets.add(facet)
                        node_counts["Facet"] += 1
                    else:
                        facet_node = self.graph.nodes.match("Facet", name=facet).first()
                    self.graph.create(Relationship(article_node, "RELATES_TO", facet_node))
                    relationship_counts["RELATES_TO"] += 1
                    actions = self.get_compliance_actions(article_body, facet)
                    for action in actions:
                        action_node = Node("ComplianceAction", category=action["category"], description=action["description"], facet=facet, article=article_title)
                        self.graph.create(action_node)
                        self.graph.create(Relationship(article_node, "RECOMMENDS", action_node))
                        self.graph.create(Relationship(action_node, "SUPPORTS", facet_node))

    def map_project_requirements(self, requirements_text: str):
        matched_facets = self.extract_responsible_ai_facet(requirements_text)
        for facet in matched_facets:
            print(f"\nRequirement maps to facet: {facet}")
            facet_node = self.graph.nodes.match("Facet", name=facet).first()
            if facet_node:
                rels = self.graph.match((None, facet_node), r_type="RELATES_TO")
                seen_titles = set()
                consolidated_actions = []
                for rel in rels:
                    article_node = rel.start_node
                    title = article_node.get("title", "Unknown Article")
                    if title not in seen_titles:
                        snippet = article_node["text"][:200].replace('\n', ' ')
                        print(f"  Suggested guideline from {title}: {snippet}...")
                        # Look up compliance actions from the knowledge graph for this article and facet
                        action_rels = self.graph.match((article_node, None), r_type="RECOMMENDS")
                        for action_rel in action_rels:
                            action_node = action_rel.end_node
                            consolidated_actions.append({
                                "category": action_node.get("category", "Unknown"),
                                "description": action_node.get("description", "No description")
                            })
                        seen_titles.add(title)
                if consolidated_actions:
                    print(f"    Consolidated Compliance Actions for facet '{facet}':")
                    for action in consolidated_actions:
                        print(f"      📋 {action['category']}: {action['description']}")

    def print_graph(self):
        node_count = self.graph.evaluate("MATCH (n) RETURN count(n)")
        rel_count = self.graph.evaluate("MATCH ()-[r]->() RETURN count(r)")
        print(f"\n✅ Graph built successfully:")
        print(f"  🔹 Total Nodes: {node_count}")
        print(f"  🔹 Total Relationships: {rel_count}")
        facets = list(self.facet_descriptions.keys())
        for facet in facets:
            query = f"""
            MATCH (a:Article)-[:RELATES_TO]->(f:Facet {{name: '{facet}'}})
            RETURN a.title AS Article, a.text AS Snippet
            """
            results = self.graph.run(query).data()
            if results:
                import pandas as pd
                df = pd.DataFrame(results)
                df['Snippet'] = df['Snippet'].apply(lambda x: x[:120].replace('\n', ' ') + "..." if len(x) > 120 else x)
                print(f"\n--- Articles related to {facet.upper()} ---")
                print(df.to_string(index=False))
            else:
                print(f"No articles found for the facet: {facet}")

    def trigger_graph_build(self, folder_path: str = "./regulations"):
        documents = self.parse_documents(folder_path)
        self.build_knowledge_graph(documents)
