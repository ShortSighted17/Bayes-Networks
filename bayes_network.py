from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from enum import Enum
from parser import GraphData, Edge
import itertools


class Weather(Enum):
    MILD = 'mild'
    STORMY = 'stormy'
    EXTREME = 'extreme'


@dataclass
class BayesNode:
    name: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    domain: List[str] = field(default_factory=lambda: ['true', 'false'])


class BayesianNetwork:

    def __init__(self, graph_data: GraphData):
        self.graph = graph_data
        self.nodes: Dict[str, BayesNode] = {}
        
        # Build the network structure
        self._build_network()
    
    def _build_network(self):
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
        idx = {'mild': 0, 'stormy': 1, 'extreme': 2}
        return self.graph.weather_prior[idx[weather]]
    
    def get_flooding_prob(self, edge_id: int, flooded: bool, weather: str) -> float:
        edge = self.graph.edges[edge_id]
        prob_flooded = edge.flood_prob_given_weather(weather)
        
        if flooded:
            return prob_flooded
        else:
            return 1.0 - prob_flooded
    
    def get_evacuees_prob(self, vertex: int, has_evacuees: bool, 
                          flooded_edges: Dict[int, bool]) -> float:
        # Get incident edges for this vertex
        incident_edge_ids = self.graph.vertex_edges.get(vertex, [])
        
        prob_no_evacuees = 1.0
        
        for eid in incident_edge_ids:
            if flooded_edges.get(eid, False):  # Edge is flooded
                edge = self.graph.edges[eid]
                # qi = min(1, P1 / weight)
                qi = min(1.0, self.graph.p1 / edge.weight)
                prob_no_evacuees *= qi
        
        prob_evacuees = 1.0 - prob_no_evacuees
        
        if has_evacuees:
            return prob_evacuees
        else:
            return prob_no_evacuees
    
    # =====================================================
    # DISPLAY METHODS
    # =====================================================
    
    def print_network_structure(self):
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
