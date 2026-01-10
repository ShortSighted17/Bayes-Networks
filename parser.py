from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Edge:
    edge_id: int
    u: int
    v: int
    weight: int
    flood_prob_mild: float  # P(Flooded | mild weather)
    
    def flood_prob_given_weather(self, weather: str) -> float:
        if weather == 'mild':
            return self.flood_prob_mild
        elif weather == 'stormy':
            return min(1.0, 2 * self.flood_prob_mild)
        elif weather == 'extreme':
            return min(1.0, 3 * self.flood_prob_mild)
        else:
            raise ValueError(f"Unknown weather type: {weather}")


@dataclass
class GraphData:
    num_vertices: int = 0
    edges: Dict[int, Edge] = field(default_factory=dict)
    p1: float = 0.3  # Default value for noisy-or parameter
    weather_prior: Tuple[float, float, float] = (0.333, 0.333, 0.334)  # Default: uniform-ish
    vertex_edges: Dict[int, List[int]] = field(default_factory=dict)
    
    def get_vertices(self) -> List[int]:
        return list(range(1, self.num_vertices + 1))
    
    def get_incident_edges(self, vertex: int) -> List[Edge]:
        edge_ids = self.vertex_edges.get(vertex, [])
        return [self.edges[eid] for eid in edge_ids]


def parse_file(file_path: str) -> GraphData:
    data = GraphData()
    
    with open(file_path, 'r') as f:
        for line in f:
            # strip comments
            line = line.split(';')[0].strip()
            if not line:
                continue
            
            parts = line.split()
            
            # Parse number of vertices: #V 4
            if line.startswith('#V '):
                data.num_vertices = int(parts[1])
                # Initialize vertex_edges for all vertices
                for v in range(1, data.num_vertices + 1):
                    data.vertex_edges[v] = []
            
            # Parse P1 parameter: #P1 0.3
            elif line.startswith('#P1'):
                data.p1 = float(parts[1])
            
            # Parse weather prior: #W 0.1 0.4 0.5
            elif line.startswith('#W '):
                p_mild = float(parts[1])
                p_stormy = float(parts[2])
                p_extreme = float(parts[3])
                data.weather_prior = (p_mild, p_stormy, p_extreme)
            
            # Parse edge: #E1 1 3 W1 F 0.2
            elif line.startswith('#E'):
                edge_id = int(parts[0][2:])  # Extract number after #E
                u = int(parts[1])
                v = int(parts[2])
                
                # Parse weight
                weight = 1  # Default weight
                flood_prob = 0.0  # Default: not flooded
                
                for i, tok in enumerate(parts[3:], start=3):
                    if tok.startswith('W'):
                        weight = int(tok[1:])
                    elif tok == 'F':
                        # Next token should be the probability
                        if i + 1 < len(parts):
                            flood_prob = float(parts[i + 1])
                
                edge = Edge(edge_id, u, v, weight, flood_prob)
                data.edges[edge_id] = edge
                
                # Record which edges are incident to which vertices
                if u not in data.vertex_edges:
                    data.vertex_edges[u] = []
                if v not in data.vertex_edges:
                    data.vertex_edges[v] = []
                data.vertex_edges[u].append(edge_id)
                data.vertex_edges[v].append(edge_id)
    
    return data


def print_graph_data(data: GraphData):
    print(f"Number of vertices: {data.num_vertices}")
    print(f"P1 parameter: {data.p1}")
    print(f"Weather prior: mild={data.weather_prior[0]}, "
          f"stormy={data.weather_prior[1]}, extreme={data.weather_prior[2]}")
    print(f"\nEdges:")
    for eid, edge in sorted(data.edges.items()):
        print(f"  E{eid}: {edge.u} -- {edge.v}, weight={edge.weight}, "
              f"P(flood|mild)={edge.flood_prob_mild}")
    print(f"\nVertex-Edge incidence:")
    for v in data.get_vertices():
        edges = data.vertex_edges.get(v, [])
        print(f"  V{v}: incident edges = {edges}")


# Test code - runs when this file is executed directly
if __name__ == "__main__":
    # Create a test input file
    test_input = """
#V 4          ; number of vertices n in graph (from 1 to n)
#P1 0.3       ; Value of parameter P1

#E1 1 3 W1 F 0.2  ; Edge1 between vertices 1 and 3, weight 1, flooded probability given mild weather 0.2
#E2 2 3 W3 F 0.1  ; Edge2 between vertices 2 and 3, weight 3, flooded probability given mild weather 0.1
#E3 2 4 W3 F 0.3  ; Edge3 between vertices 2 and 4, weight 3, flooded probability given mild weather 0.3
#E4 3 4 W4 F 0    ; Edge4 between vertices 3 and 4, weight 4

#W 0.1 0.4 0.5 ; Prior distribution over weather: 0.1 for mild, 0.4 for stormy, 0.5 for extreme
"""
    
    # Write test file
    with open("test_input.txt", "w") as f:
        f.write(test_input)
    
    # Parse and display
    data = parse_file("test_input.txt")
    print_graph_data(data)
