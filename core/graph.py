import os
import json
import uuid
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
import pickle

from config.settings import ConfigManager
from prompts.templates import GraphPrompts
from utils.logging import get_logger
from utils.aux import parse_json_response
from langchain.schema import AIMessage, SystemMessage, HumanMessage

class EntityGraph:
    """Entity graph for tracking conversation information"""
    
    def __init__(
        self,
        target: str,
        graph_model,
        conv_model,
        routine: Optional[str] = None,
        working_directory: Optional[str] = None,
        language: str = "English",
        **params
    ):
        self.config = ConfigManager()
        self.prompts = GraphPrompts()
        self.target = target
        self.graph_model = graph_model
        self.conv_model = conv_model
        self.routine = routine
        self.working_directory = working_directory
        self.language = language

        # Graph parameters
        self.node_hit_threshold = params.get('node_hit_threshold', np.inf)
        self.confidential_threshold = params.get('confidential_threshold', 0.2)
        self.relevance_threshold = params.get('relevance_threshold', 0.2)
        self.weight_threshold = params.get('weight_threshold', 0.8)
        self.alpha = params.get('alpha', 10.0)
        self.beta = params.get('beta', 1.0)
        self.gamma = params.get('gamma', 1.0)
        
        self.step = 0
        self.accomplish = False
        self.prev_node = None
        self.logger = get_logger(self.__class__.__name__)
        
        self._ensure_working_directory()
    
    def _ensure_working_directory(self):
        """Ensure working directory exists"""
        if self.working_directory and not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
    
    def init(self, save: bool = False):
        """Initialize the graph"""
        self._initialize_graph()
        self._clustering()
        if save:
            self.save_graphs(self.working_directory)
    
    def save_graphs(self, output_dir: str):
        """Save entity and relation graphs to files"""
        if not self.working_directory:
            return
            
        entity_graph_file = os.path.join(output_dir, "entity_graph.pkl")
        relation_graph_file = os.path.join(output_dir, "relation_graph.pkl")
        
        with open(entity_graph_file, "wb") as f:
            pickle.dump(self.entity_graph, f)
        with open(relation_graph_file, "wb") as f:
            pickle.dump(self.relation_graph, f)
            
        self.logger.info(f"Saved graphs to {self.working_directory}")
    
    def load_graphs(self, entity_graph_path: str, relation_graph_path: str):
        """Load graphs from files"""
        if not os.path.exists(entity_graph_path):
            raise FileNotFoundError(f"Entity graph not found: {entity_graph_path}")
        if not os.path.exists(relation_graph_path):
            raise FileNotFoundError(f"Relation graph not found: {relation_graph_path}")
            
        with open(entity_graph_path, "rb") as f:
            self.entity_graph = pickle.load(f)
        with open(relation_graph_path, "rb") as f:
            self.relation_graph = pickle.load(f)
            
        self._clustering()
        self.logger.info("Loaded graphs successfully")
    
    def _initialize_graph(self):
        """Initialize entity and relation graphs using LLM"""
        # Step 1: Retrieve entities
        entities = self._retrieve_entities()
        
        # Step 2: Initialize entity attributes
        nodes = self._initialize_entity_attributes(entities)
        
        # Step 3: Create entity graph edges
        entity_edges = self._create_entity_edges(entities)
        
        # Step 4: Create relation graph edges
        relation_edges = self._create_relation_edges(entities)
        
        # Build graphs
        self.entity_graph = self._build_graph(nodes, entity_edges)
        self.relation_graph = self._build_graph(nodes, relation_edges)
        
        # Initialize node states
        self._initialize_node_states()
        
    def _retrieve_entities(self) -> List[Dict[str, str]]:
        """Retrieve entities needed for the target"""
        messages = []
        entities = []
        
        prompt = self.prompts.get("ENTITY_RETRIEVE", purpose=self.target, language=self.language)
        if self.routine:
            routine_prompt = self.prompts.get("ROUTINE_ADDITION", routine=self.routine)
            prompt += "\n" + routine_prompt
            
        messages.append(SystemMessage(content=prompt))
        response = self.graph_model.invoke(messages)
        
        try:
            result = parse_json_response(response.content)
            entities.extend(result.get("entities", []))
            messages.append(response)
            
            # Continue if needed
            endpoint = result.get("endpoint", True)
            if isinstance(endpoint, str):
                endpoint = endpoint.lower() == "true"
            elif isinstance(endpoint, bool):
                endpoint = endpoint
            else:
                raise ValueError(f"Unexpected endpoint type: {type(endpoint)}")
            iteration = 1
            
            while not endpoint and iteration < 10:
                messages.append(HumanMessage(content=self.prompts.get("CONTINUE_ENTITY_RETRIEVE")))
                response = self.graph_model.invoke(messages)
                result = parse_json_response(response.content)
                
                new_entities = result.get("entities", [])
                if not new_entities:
                    break
                    
                entities.extend(new_entities)
                messages.append(response)
                endpoint = result.get("endpoint", True)
                if isinstance(endpoint, str):
                    endpoint = endpoint.lower() == "true"
                elif isinstance(endpoint, bool):
                    endpoint = endpoint
                else:
                    raise ValueError(f"Unexpected endpoint type: {type(endpoint)}")
                iteration += 1
                
        except json.JSONDecodeError as e:
            self.logger.error(f"response.content: {response.content}")
            self.logger.error(f"Failed to parse entity response: {e}")
            raise
        
        # Assign IDs to entities
        return [{"id": f"v{i}", "name": entity} for i, entity in enumerate(entities, start=1)]
    
    def _initialize_entity_attributes(self, entities: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Initialize attributes for entities"""
        nodes = []
        chunk_size = 10
        
        for i in range(0, len(entities), chunk_size):
            chunk = entities[i:i + chunk_size]
            entities_str = ", ".join([f"id: {e['id']}, name: {e['name']}" for e in chunk])

            prompt = self.prompts.get("INIT_GRAPH_ENTITY", purpose=self.target, entities=entities_str, language=self.language)
                
            response = self.graph_model.invoke([HumanMessage(content=prompt)])
            
            try:
                chunk_nodes = parse_json_response(response.content)
                nodes.extend(chunk_nodes)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse entity attributes: {e}")
                raise
        print(f"the total number of nodes: {len(nodes)}")    
        return nodes
    
    def _create_entity_edges(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create edges for entity graph (dependencies)"""
        return self._create_edges(entities, "INIT_ENTITY_GRAPH_EDGES", "CONTINUE_INIT_ENTITY_GRAPH_EDGES")
    
    def _create_relation_edges(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create edges for relation graph"""
        return self._create_edges(entities, "INIT_RELATION_GRAPH_EDGES", "CONTINUE_INIT_RELATION_GRAPH_EDGES")
    
    def _create_edges(self, entities: List[Dict[str, str]], init_prompt_key: str, continue_prompt_key: str) -> List[Dict[str, str]]:
        """Generic edge creation method"""
        messages = []
        edges = []
        iteration = 0
        endpoint = False
        entities_str = ", ".join([f"id: {e['id']}, name: {e['name']}" for e in entities])
        
        while not endpoint and iteration < 10:
            if iteration == 0:
                prompt = self.prompts.get(init_prompt_key, purpose=self.target, entities=entities_str, language=self.language)
            else:
                prompt = self.prompts.get(continue_prompt_key)
                
            messages.append(HumanMessage(content=prompt))
            response = self.graph_model.invoke(messages)
            
            try:
                result = parse_json_response(response.content)
                new_edges = result.get("edges", [])
                
                if not new_edges:
                    break
                    
                edges.extend(new_edges)
                endpoint = result.get("endpoint", True)
                if isinstance(endpoint, str):
                    endpoint = endpoint.lower() == "true"
                elif isinstance(endpoint, bool):
                    endpoint = endpoint
                else:
                    raise ValueError(f"Unexpected endpoint type: {type(endpoint)}")
                messages.append(response)
                iteration += 1
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse edges: {e}")
                break
                
        return edges
    
    def _build_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, str]]) -> nx.DiGraph:
        """Build NetworkX graph from nodes and edges"""
        G = nx.DiGraph()
        
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                G.add_node(node_id, **node)
                
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target and source in G and target in G:
                G.add_edge(source, target, **edge)
                
        self.logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _initialize_node_states(self):
        """Initialize node states (value, hit, status)"""
        for node in self.entity_graph.nodes:
            if "value" not in self.entity_graph.nodes[node]:
                self.entity_graph.nodes[node]["value"] = ""
            if "hit" not in self.entity_graph.nodes[node]:
                self.entity_graph.nodes[node]["hit"] = 0
            if "status" not in self.entity_graph.nodes[node]:
                self.entity_graph.nodes[node]["status"] = 0
    
    def _clustering(self):
        """Perform community detection on the graph"""
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            self.logger.warning("leidenalg/igraph not installed; assigning all nodes to community 0")
            for node in self.relation_graph.nodes():
                self.relation_graph.nodes[node]["community"] = 0
                self.entity_graph.nodes[node]["community"] = 0
            return
        
        # Convert to undirected graph for community detection
        ud_graph = self.relation_graph.to_undirected()
        mapping = dict(enumerate(ud_graph.nodes()))
        inv_mapping = {v: k for k, v in mapping.items()}
        
        # Check for edge weights
        has_weights = any('weight' in ud_graph.get_edge_data(u, v, {}) for u, v in ud_graph.edges())
        
        # Build igraph
        edges = [(inv_mapping[u], inv_mapping[v]) for u, v in ud_graph.edges()]
        ig_g = ig.Graph(len(mapping), edges)
        ig_g.vs["name"] = [mapping[i] for i in range(len(mapping))]
        
        if has_weights:
            weights = [ud_graph.get_edge_data(u, v).get('weight', 1.0) for u, v in ud_graph.edges()]
            ig_g.es['weight'] = weights
            partition = leidenalg.find_partition(ig_g, leidenalg.RBConfigurationVertexPartition, weights='weight')
        else:
            partition = leidenalg.find_partition(ig_g, leidenalg.RBConfigurationVertexPartition)
        
        # Assign communities
        for comm_id, community in enumerate(partition):
            for vid in community:
                node_name = ig_g.vs[vid]["name"]
                self.relation_graph.nodes[node_name]["community"] = comm_id
                self.entity_graph.nodes[node_name]["community"] = comm_id
                
        self.logger.info("Community detection completed")
    
    def get_hint_message(self) -> Tuple[str, bool]:
        """Generate hint message for next conversation turn"""
        selection = self._select_node()
        
        if selection is None:
            # All information collected, generate final hint
            prompt = self.prompts.get(
                "HINT_MESSAGE_ACCOMPLISH",
                collected=self._serialize_nodes_with_value(self.entity_graph),
                purpose=self.target, 
                language=self.language
            )
            response = self.graph_model.invoke([SystemMessage(content=prompt)])
            hint_message = response.content
            self.accomplish = True
            self.logger.info("Generated accomplishment hint")
        else:
            # Generate hint for next node
            node_id, node_data = selection
            prompt = self.prompts.get(
                "HINT_MESSAGE_RETRIEVE",
                collected=self._serialize_nodes_with_value(self.entity_graph),
                recommendation=self._serialize_node_info(node_data),
                purpose=self.target, 
                language=self.language
            )
            response = self.graph_model.invoke([SystemMessage(content=prompt)])
            hint_message = response.content
            self.logger.info(f"Generated hint for node {node_id}")
        
        return hint_message, self.accomplish
    
    def accept_message(self, hint_message: str, query_message: str, user_message: str):
        """Process user message and update graph"""
        updated_nodes, new_nodes = self._process_user_message(hint_message, query_message, user_message)
        self._update_graph(updated_nodes, new_nodes)
        
    def _get_available_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get nodes available for querying"""
        available = []
        
        for node_id, data in self.entity_graph.nodes(data=True):
            # Check status
            if data.get("status", 0) not in (0, 1):
                continue
                
            # Check hit threshold
            if data.get("hit", 0) >= self.node_hit_threshold:
                continue
                
            # Check weight threshold
            if data.get("weight", 0.0) < self.weight_threshold:
                continue
                
            # Check prerequisites
            prerequisites_met = True
            for pred in self.entity_graph.predecessors(node_id):
                if self.entity_graph.nodes[pred].get("status", 0) != 2:
                    prerequisites_met = False
                    break
                    
            if prerequisites_met:
                available.append((node_id, data))
                
        return available
    
    def _select_node(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Select next node to query using scoring algorithm"""
        available_nodes = self._get_available_nodes()
        if not available_nodes:
            return None
        
        # Calculate PageRank
        pr = nx.pagerank(self.relation_graph)
        
        # Calculate scores for each node
        best_node_id = None
        best_data = None
        best_score = -float('inf')
        
        weights, entropies, topologies, communities = [], [], [], []
        
        for nid, data in available_nodes:
            weights.append(data.get("weight", 0.0))
            entropies.append(data.get("uncertainty", 1.0))
            topologies.append(pr.get(nid, 0))
            
            if self.prev_node is None:
                communities.append(0.0)
            else:
                communities.append(self._calculate_community_score(nid, data))
        
        # Normalize scores
        normalize = lambda x: (np.array(x) - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) - np.min(x) != 0 else np.zeros_like(x)
        
        weights = normalize(weights)
        entropies = normalize(entropies)
        importance_score = normalize(weights * 5 + entropies)  # Weight more heavily
        topologies = normalize(topologies)
        communities = normalize(communities)
        
        # Calculate combined scores
        scores = self.alpha * importance_score + self.beta * topologies + self.gamma * communities
        
        # Select best node
        best_index = np.argmax(scores)
        best_node_id = available_nodes[best_index][0]
        best_data = available_nodes[best_index][1]
        best_score = scores[best_index]
        
        # Log scores
        for i, (nid, data) in enumerate(available_nodes):
            self.logger.debug(
                f"Node {nid}: weight={weights[i]:.3f}, entropy={entropies[i]:.3f}, "
                f"topology={topologies[i]:.3f}, community={communities[i]:.3f}, score={scores[i]:.3f}"
            )
        
        # Update hit counter
        self.entity_graph.nodes[best_node_id]["hit"] += 1
        self.prev_node = best_node_id
        
        self.logger.info(f"Selected node {best_node_id} with score {best_score:.3f}")
        return best_node_id, best_data
    
    def _calculate_community_score(self, cand_id: str, cand_data: Dict[str, Any]) -> float:
        """Calculate community coherence score"""
        if self.prev_node is None:
            return 1.0
            
        prev_comm = self.entity_graph.nodes[self.prev_node].get("community", None)
        if prev_comm is None:
            return 1.0
        
        # Get candidate neighbors
        candidate_neighbors = set(self.entity_graph.neighbors(cand_id)).union(
            set(self.relation_graph.predecessors(cand_id))
        )
        
        d_in = 0
        d_out = 0
        
        for neighbor in candidate_neighbors:
            neighbor_comm = self.entity_graph.nodes[neighbor].get("community", None)
            if neighbor_comm == prev_comm:
                d_in += 1
            else:
                d_out += 1
        
        epsilon = 1e-9
        eta = d_in / (d_in + d_out + epsilon)
        
        candidate_comm = cand_data.get("community", None)
        if candidate_comm is not None and candidate_comm == prev_comm:
            score = math.exp(eta)
        else:
            score = math.exp(-eta)
            
        return score
    
    def _process_user_message(self, hint_message: str, query_message: str, human_message: str) -> Tuple[List[str], List[str]]:
        """Extract information from user message and update graph"""
        messages = []
        extract_info = {"exist_nodes": [], "new_nodes": []}
        iteration = 0
        endpoint = False
        
        while not endpoint and iteration < 10:
            if iteration == 0:
                prompt = self.prompts.get(
                    "EXTRACT_INFO",
                    purpose=self.target,
                    graph=self._serialize_nodes(self.entity_graph),
                    hint_message=hint_message,
                    query_message=query_message,
                    human_message=human_message,
                    language=self.language
                )
            else:
                prompt = self.prompts.get("CONTINUE_EXTRACT_INFO")
                
            messages.append(HumanMessage(content=prompt))
            response = self.conv_model.invoke(messages)
            
            try:
                result = parse_json_response(response.content)
                extract_info["exist_nodes"].extend(result.get("exist_nodes", []))
                extract_info["new_nodes"].extend(result.get("new_nodes", []))
                
                messages.append(response)
                endpoint = result.get("endpoint", True)
                if isinstance(endpoint, str):
                    endpoint = endpoint.lower() == "true"
                elif isinstance(endpoint, bool):
                    endpoint = endpoint
                else:
                    raise ValueError(f"Unexpected endpoint type: {type(endpoint)}")
                iteration += 1
                
                # Break if no new information
                if not result.get("exist_nodes") and not result.get("new_nodes"):
                    break
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse extraction response: {e}")
                break
        
        # Update existing nodes
        updated_nodes = []
        for entry in extract_info.get("exist_nodes", []):
            node_id = entry.get("id")
            value = str(entry.get("value", "")).strip()
            confidential_level = entry.get("confidential_level", 0.0)
            
            if not value or node_id not in self.entity_graph.nodes:
                continue
                
            updated_nodes.append(node_id)
            
            # Store value history
            if "value_history" not in self.entity_graph.nodes[node_id]:
                self.entity_graph.nodes[node_id]["value_history"] = []
            self.entity_graph.nodes[node_id]["value_history"].append(
                self.entity_graph.nodes[node_id].get("value", "")
            )
            
            # Update node
            self.entity_graph.nodes[node_id]["value"] = value
            self.entity_graph.nodes[node_id]["confidential_level"] = confidential_level
            
            if confidential_level >= self.confidential_threshold:
                self.entity_graph.nodes[node_id]["status"] = 2
            else:
                self.entity_graph.nodes[node_id]["status"] = 1
                
            self.logger.info(f"Updated node {node_id} with value: {value[:50]}...")
        
        # Add new nodes
        new_nodes = []
        for entry in extract_info.get("new_nodes", []):
            node_id = entry.get("id", f"v{uuid.uuid4().hex[:8]}")
            name = entry.get("name", "")
            value = str(entry.get("value", "")).strip()
            relevance = entry.get("relevance", 0.0)
            confidential_level = entry.get("confidential_level", 0.0)
            
            if relevance < self.relevance_threshold:
                self.logger.info(f"Skipping node {name} with low relevance: {relevance:.2f}")
                continue
                
            if not name or not value:
                continue
                
            new_node_data = {
                "id": node_id,
                "name": name,
                "description": entry.get("description", ""),
                "value": value,
                "weight": entry.get("weight", 1.0),
                "uncertainty": entry.get("uncertainty", 1.0),
                "confidential_level": confidential_level,
                "status": 2 if confidential_level >= self.confidential_threshold else 1,
                "hit": 1,
                "community": 0  # Will be updated in clustering
            }
            
            self.entity_graph.add_node(node_id, **new_node_data)
            self.relation_graph.add_node(node_id, **new_node_data)
            new_nodes.append(node_id)
            
            self.logger.info(f"Added new node {node_id}: {name}")
        
        return updated_nodes, new_nodes
    
    def _update_graph(self, updated_nodes: List[str], new_nodes: List[str]):
        """Update graph structure and weights based on new information"""
        if not updated_nodes:
            return
            
        # Get neighbors of updated nodes
        all_neighbors = []
        for node_id in updated_nodes:
            neighbors = list(self.entity_graph.neighbors(node_id))
            neighbors = [n for n in neighbors if self.entity_graph.nodes[n]["status"] in (0, 1)]
            all_neighbors.extend(neighbors)
        
        if not all_neighbors:
            return
            
        # Update weights and uncertainties in chunks
        chunk_size = 20
        for i in range(0, len(all_neighbors), chunk_size):
            chunk = all_neighbors[i:i + chunk_size]
            
            relevant_nodes = "\n".join([
                f"node {self.entity_graph.nodes[n]['id']}, {self.entity_graph.nodes[n]['name']}, "
                f"initial weight {self.entity_graph.nodes[n]['weight']}, "
                f"initial uncertainty {self.entity_graph.nodes[n]['uncertainty']}"
                for n in chunk
            ])
            
            prompt = self.prompts.get(
                "UPDATE_GRAPH",
                purpose=self.target,
                collected=self._serialize_nodes_with_value(self.entity_graph),
                relevant_nodes=relevant_nodes
            )
            
            response = self.graph_model.invoke([SystemMessage(content=prompt)])
            
            try:
                updates = parse_json_response(response.content)
                for update in updates:
                    node_id = update.get("id")
                    if node_id in self.entity_graph.nodes:
                        self.entity_graph.nodes[node_id]["weight"] = update.get("weight", 
                            self.entity_graph.nodes[node_id]["weight"])
                        self.entity_graph.nodes[node_id]["uncertainty"] = update.get("uncertainty",
                            self.entity_graph.nodes[node_id]["uncertainty"])
                        self.logger.info(f"Updated node {node_id}: {update.get('update_reason', 'No reason')}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse update response: {e}")
        
        # Re-cluster if new nodes added
        if new_nodes:
            self._clustering()
    
    def _serialize_nodes(self, graph: nx.DiGraph) -> str:
        """Serialize graph nodes for prompts"""
        serialized_nodes = []
        serialize_keys = ["id", "name", "description"]
        
        for node, attrs in graph.nodes(data=True):
            attr_str = ", ".join(f"{key}: {attrs.get(key, '')}" for key in serialize_keys)
            serialized_nodes.append(attr_str)
            
        return "\n".join(serialized_nodes)
    
    def _serialize_nodes_with_value(self, graph: nx.DiGraph) -> str:
        """Serialize nodes with values for prompts"""
        serialized_nodes = []
        serialize_keys = ["name", "description", "value"]
        
        for node, attrs in graph.nodes(data=True):
            if attrs.get("value"):
                attr_str = ", ".join(f"{key}: {attrs.get(key, '')}" for key in serialize_keys)
                if attrs.get("status", 0) == 1:
                    attr_str += ", this value is with low confidence"
                serialized_nodes.append(attr_str)
                
        return "\n".join(serialized_nodes)
    
    def _serialize_node_info(self, node: Dict[str, Any]) -> str:
        """Serialize single node information"""
        info = f"- Name: {node.get('name', '')} (a short name of the entity)\n"
        info += f"- Description: {node.get('description', '')} (a detailed description of the node)\n"
        info += f"- Confidential level: {node.get('confidential_level', 0)} (confidence level [0, 1])\n"
        info += f"- Value: {node.get('value', '')} (the extracted value)\n"
        info += f"- Hit: {node.get('hit', 0)} (number of times queried)\n"
        info += f"- Status: {node.get('status', 0)} (0 for unknown, 1 for low confidence, 2 for high confidence)\n"
        return info