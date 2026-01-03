r"""
Bayesian Network for Hurricane Evacuation Problem

This module defines the structure of the Bayesian Network:

NETWORK STRUCTURE:
==================
    
         Weather (W)
        /    |    \
       v     v     v
    F(e1)  F(e2)  F(e3) ...  (Flooding variables)
       \     |     /
        v    v    v
       Ev(v1) Ev(v2) ...     (Evacuees variables)

The network has three types of nodes:
1. Weather (W): Root node with 3 states {mild, stormy, extreme}
2. Flooding F(e): One for each edge, depends only on Weather
3. Evacuees Ev(v): One for each vertex, depends on flooding of incident edges

CONDITIONAL PROBABILITY TABLES:
===============================

1. P(Weather): Prior distribution given in input file

2. P(F(e) | Weather): For each edge e
   - P(F(e)=true | mild) = base probability from input
   - P(F(e)=true | stormy) = min(1, 2 * base)
   - P(F(e)=true | extreme) = min(1, 3 * base)

3. P(Ev(v) | F(e1), F(e2), ...): Uses Noisy-OR model
   
   Noisy-OR Model Explanation:
   ---------------------------
   The noisy-OR is a compact way to represent CPTs when multiple causes 
   can independently produce an effect.
   
   Each cause i (flooded edge) has a probability qi of producing the effect
   (evacuees present) when that cause is present.
   
   The noisy-OR formula:
   P(effect=false | causes) = ∏(1 - qi) for all present causes
   P(effect=true | causes) = 1 - P(effect=false | causes)
   
   In our case: qi = min(1, P1 / weight(e))
   where P1 is a global parameter and weight(e) is the edge weight.
   
   This means:
   - Lower weight edges have HIGHER probability of causing evacuees
   - If NO edges are flooded, P(Evacuees) = 0 (no leak probability)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from enum import Enum
from parser import GraphData, Edge
import itertools


class Weather(Enum):
    """Weather states"""
    MILD = 'mild'
    STORMY = 'stormy'
    EXTREME = 'extreme'


@dataclass
class BayesNode:
    """
    Represents a node in the Bayesian Network.
    
    Attributes:
        name: Unique identifier for the node (e.g., "W", "F1", "Ev1")
        parents: List of parent node names
        children: List of child node names
        domain: Possible values this variable can take
    """
    name: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    domain: List[str] = field(default_factory=lambda: ['true', 'false'])


class BayesianNetwork:
    """
    Bayesian Network for the Hurricane Evacuation Problem.
    
    The network structure:
    - Weather (W) is the root node
    - Each edge has a Flooding node F(e) with Weather as parent
    - Each vertex has an Evacuees node Ev(v) with incident flooding nodes as parents
    """
    
    def __init__(self, graph_data: GraphData):
        """
        Construct the Bayesian Network from parsed graph data.
        
        Args:
            graph_data: Parsed graph structure and parameters
        """
        self.graph = graph_data
        self.nodes: Dict[str, BayesNode] = {}
        
        # Build the network structure
        self._build_network()
    
    def _build_network(self):
        """Construct all nodes and their parent-child relationships."""
        
        # 1. Create Weather node (root)
        weather_node = BayesNode(
            name="W",
            parents=[],
            children=[],
            domain=['mild', 'stormy', 'extreme']
        )
        self.nodes["W"] = weather_node
        
        # 2. Create Flooding nodes for each edge
        for edge_id in self.graph.edges:
            flood_name = f"F{edge_id}"
            flood_node = BayesNode(
                name=flood_name,
                parents=["W"],  # Weather is the only parent
                children=[],
                domain=['true', 'false']
            )
            self.nodes[flood_name] = flood_node
            weather_node.children.append(flood_name)
        
        # 3. Create Evacuees nodes for each vertex
        for v in self.graph.get_vertices():
            evac_name = f"Ev{v}"
            
            # Parents are the flooding nodes of incident edges
            incident_edges = self.graph.vertex_edges.get(v, [])
            parent_flood_nodes = [f"F{eid}" for eid in incident_edges]
            
            evac_node = BayesNode(
                name=evac_name,
                parents=parent_flood_nodes,
                children=[],
                domain=['true', 'false']
            )
            self.nodes[evac_name] = evac_node
            
            # Update children lists of parent flood nodes
            for parent in parent_flood_nodes:
                self.nodes[parent].children.append(evac_name)
    
    # =====================================================
    # CONDITIONAL PROBABILITY TABLE (CPT) CALCULATIONS
    # =====================================================
    
    def get_weather_prob(self, weather: str) -> float:
        """
        P(Weather = w)
        
        Returns the prior probability of a weather state.
        """
        idx = {'mild': 0, 'stormy': 1, 'extreme': 2}
        return self.graph.weather_prior[idx[weather]]
    
    def get_flooding_prob(self, edge_id: int, flooded: bool, weather: str) -> float:
        """
        P(F(e) = flooded | Weather = w)
        
        Returns the conditional probability of flooding given weather.
        
        The probability of flooding increases with worse weather:
        - Stormy: 2x mild probability
        - Extreme: 3x mild probability
        """
        edge = self.graph.edges[edge_id]
        prob_flooded = edge.flood_prob_given_weather(weather)
        
        if flooded:
            return prob_flooded
        else:
            return 1.0 - prob_flooded
    
    def get_evacuees_prob(self, vertex: int, has_evacuees: bool, 
                          flooded_edges: Dict[int, bool]) -> float:
        """
        P(Ev(v) = has_evacuees | flooding states of incident edges)
        
        Uses the Noisy-OR model.
        
        The Noisy-OR model works as follows:
        1. Each flooded edge e incident to vertex v can independently 
           cause evacuees to be present with probability qi
        2. qi = min(1, P1 / weight(e))
        3. P(no evacuees) = product of (1 - qi) for all flooded edges
        4. P(evacuees) = 1 - P(no evacuees)
        
        If no edges are flooded, P(evacuees) = 0 (this is a property
        of noisy-OR with no "leak" probability)
        
        Args:
            vertex: The vertex ID
            has_evacuees: True if we want P(Ev=true), False for P(Ev=false)
            flooded_edges: Dict mapping edge_id -> is_flooded for incident edges
        """
        # Get incident edges for this vertex
        incident_edge_ids = self.graph.vertex_edges.get(vertex, [])
        
        # Calculate probability of NO evacuees using noisy-OR
        # P(Ev=false) = ∏(1 - qi) for all flooded incident edges
        prob_no_evacuees = 1.0
        
        for eid in incident_edge_ids:
            if flooded_edges.get(eid, False):  # Edge is flooded
                edge = self.graph.edges[eid]
                # qi = min(1, P1 / weight)
                qi = min(1.0, self.graph.p1 / edge.weight)
                prob_no_evacuees *= (1.0 - qi)
        
        prob_evacuees = 1.0 - prob_no_evacuees
        
        if has_evacuees:
            return prob_evacuees
        else:
            return prob_no_evacuees
    
    # =====================================================
    # DISPLAY METHODS
    # =====================================================
    
    def print_network_structure(self):
        """Print the structure of the Bayesian Network."""
        print("=" * 60)
        print("BAYESIAN NETWORK STRUCTURE")
        print("=" * 60)
        
        print("\nNodes and their relationships:")
        for name, node in sorted(self.nodes.items()):
            print(f"\n  {name}:")
            print(f"    Domain: {node.domain}")
            print(f"    Parents: {node.parents if node.parents else 'None (root)'}")
            print(f"    Children: {node.children if node.children else 'None (leaf)'}")
    
    def print_cpts(self):
        """
        Print all Conditional Probability Tables.
        
        This outputs the full probability distributions as required
        by the assignment.
        """
        print("\n" + "=" * 60)
        print("CONDITIONAL PROBABILITY TABLES")
        print("=" * 60)
        
        # 1. Weather prior
        print("\nWEATHER:")
        for w in ['mild', 'stormy', 'extreme']:
            print(f"  P({w}) = {self.get_weather_prob(w)}")
        
        # 2. Flooding CPTs for each edge
        for eid, edge in sorted(self.graph.edges.items()):
            print(f"\nEDGE {eid} (connects {edge.u} -- {edge.v}, weight={edge.weight}):")
            for w in ['mild', 'stormy', 'extreme']:
                prob = self.get_flooding_prob(eid, True, w)
                print(f"  P(flooded | {w}) = {prob}")
        
        # 3. Evacuees CPTs for each vertex
        for v in self.graph.get_vertices():
            incident_edges = self.graph.vertex_edges.get(v, [])
            print(f"\nVERTEX {v} (incident edges: {incident_edges}):")
            
            if not incident_edges:
                print("  P(Evacuees) = 0  (no incident edges)")
                continue
            
            # Generate all combinations of flooding states for incident edges
            # For n incident edges, we have 2^n combinations
            n_edges = len(incident_edges)
            
            for combo in itertools.product([False, True], repeat=n_edges):
                # Create flooding state dictionary
                flooded_state = {incident_edges[i]: combo[i] for i in range(n_edges)}
                
                # Build readable condition string
                conditions = []
                for i, eid in enumerate(incident_edges):
                    state = "flooded" if combo[i] else "not flooded"
                    conditions.append(f"{state} {eid}")
                condition_str = ", ".join(conditions)
                
                # Calculate probability
                prob = self.get_evacuees_prob(v, True, flooded_state)
                print(f"  P(Evacuees | {condition_str}) = {prob:.4f}")
    
    def get_topological_order(self) -> List[str]:
        """
        Return nodes in topological order (parents before children).
        
        For our specific network:
        1. Weather first
        2. All flooding nodes
        3. All evacuees nodes
        """
        order = ["W"]
        
        # Add flooding nodes
        for eid in sorted(self.graph.edges.keys()):
            order.append(f"F{eid}")
        
        # Add evacuees nodes
        for v in self.graph.get_vertices():
            order.append(f"Ev{v}")
        
        return order


# Test code
if __name__ == "__main__":
    from parser import parse_file
    
    # Create test input
    test_input = """
#V 4
#P1 0.3

#E1 1 3 W1 F 0.2
#E2 2 3 W3 F 0.1
#E3 2 4 W3 F 0.3
#E4 3 4 W4 F 0

#W 0.1 0.4 0.5
"""
    
    with open("test_bn.txt", "w") as f:
        f.write(test_input)
    
    # Parse and build network
    data = parse_file("test_bn.txt")
    bn = BayesianNetwork(data)
    
    # Display network
    bn.print_network_structure()
    bn.print_cpts()
