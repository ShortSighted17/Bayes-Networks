"""
Main Program for Assignment 3: Bayesian Network for Hurricane Evacuation

This is the interactive main program that:
1. Reads the input file and constructs the Bayesian Network
2. Displays the network structure and CPTs
3. Allows the user to:
   - Add/reset evidence
   - Query probabilities
   - Find safest paths (bonus)

Usage:
    python main.py [input_file]
    
If no input file is provided, uses a default test file.
"""

import sys
from parser import parse_file, print_graph_data
from bayes_network import BayesianNetwork
from inference import InferenceEngine


class InteractiveSession:
    """
    Manages an interactive session for querying the Bayesian Network.
    """
    
    def __init__(self, bn: BayesianNetwork, engine: InferenceEngine):
        self.bn = bn
        self.engine = engine
        self.evidence = {}  # Current evidence
    
    def run(self):
        """Run the interactive session."""
        print("\n" + "=" * 60)
        print("BAYESIAN NETWORK INTERACTIVE SESSION")
        print("=" * 60)
        
        self.print_help()
        
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                
                if not cmd:
                    continue
                
                if cmd in ['q', 'quit', 'exit']:
                    print("Goodbye!")
                    break
                elif cmd in ['h', 'help']:
                    self.print_help()
                elif cmd in ['r', 'reset']:
                    self.reset_evidence()
                elif cmd in ['e', 'evidence']:
                    self.show_evidence()
                elif cmd.startswith('add '):
                    self.add_evidence(cmd[4:])
                elif cmd in ['all', 'posteriors']:
                    self.show_all_posteriors()
                elif cmd.startswith('p ') or cmd.startswith('prob '):
                    self.query_probability(cmd)
                elif cmd.startswith('path '):
                    self.query_path(cmd[5:])
                elif cmd in ['bn', 'network']:
                    self.bn.print_network_structure()
                elif cmd in ['cpt', 'cpts', 'tables']:
                    self.bn.print_cpts()
                elif cmd in ['best', 'safest']:
                    self.find_safest_path()
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def print_help(self):
        """Print available commands."""
        print("""
Available Commands:
-------------------
  help, h          - Show this help message
  quit, q          - Exit the program
  
Evidence Management:
  reset, r         - Clear all evidence
  evidence, e      - Show current evidence
  add <var>=<val>  - Add evidence (e.g., 'add F1=true', 'add W=stormy')
  
Queries:
  all, posteriors  - Show all posterior probabilities given current evidence
  prob <var>       - Query P(var | evidence) - shows full distribution
  prob <var>=<val> - Query P(var=val | evidence) - shows single probability
  path <e1,e2,...> - Query P(path is clear | evidence)
  
Display:
  bn, network      - Show network structure
  cpt, cpts        - Show all conditional probability tables
  
Bonus:
  best, safest     - Find safest path between two vertices

Variable Names:
  W                - Weather (values: mild, stormy, extreme)
  F<n>             - Flooding at edge n (values: true, false)
  Ev<n>            - Evacuees at vertex n (values: true, false)

Examples:
  add F1=true      - Add evidence that edge 1 is flooded
  prob F5          - Show P(F5=true) and P(F5=false) given evidence
  prob F5=true     - Show only P(F5=true) given evidence
  path 1,2,3       - Show probability that edges 1, 2, 3 are all clear
        """)
    
    def reset_evidence(self):
        """Clear all evidence."""
        self.evidence = {}
        print("Evidence cleared.")
    
    def show_evidence(self):
        """Display current evidence."""
        if self.evidence:
            print("Current evidence:")
            for var, val in self.evidence.items():
                print(f"  {var} = {val}")
        else:
            print("No evidence set.")
    
    def add_evidence(self, spec: str):
        """
        Add evidence from a specification string.
        
        Format: var=value (e.g., 'F1=true', 'W=stormy', 'Ev2=false')
        """
        try:
            parts = spec.strip().split('=')
            if len(parts) != 2:
                print("Usage: add <variable>=<value>")
                return
            
            var = parts[0].strip()
            val = parts[1].strip().lower()
            
            # Try to match variable name case-insensitively
            matched_var = None
            for node_name in self.bn.nodes:
                if node_name.lower() == var.lower():
                    matched_var = node_name
                    break
            
            if matched_var is None:
                print(f"Unknown variable: {var}")
                print(f"Valid variables: {', '.join(sorted(self.bn.nodes.keys()))}")
                return
            
            var = matched_var  # Use the correctly-cased name
            
            # Validate value
            domain = self.bn.nodes[var].domain
            if val not in domain:
                print(f"Invalid value for {var}: {val}")
                print(f"Valid values: {domain}")
                return
            
            self.evidence[var] = val
            print(f"Added evidence: {var} = {val}")
        
        except Exception as e:
            print(f"Error adding evidence: {e}")
    
    def show_all_posteriors(self):
        """Show all posterior probabilities given current evidence."""
        self.engine.print_all_posteriors(self.evidence)
    
    def query_probability(self, cmd: str):
        """Query probability of a specific variable."""
        # Extract variable name
        parts = cmd.split()
        if len(parts) < 2:
            print("Usage: prob <variable> or prob <variable>=<value>")
            return
        
        var_spec = parts[1].strip()
        
        # Check if user specified a specific value (e.g., "F5=true")
        if '=' in var_spec:
            var, requested_val = var_spec.split('=', 1)
            var = var.strip()
            requested_val = requested_val.strip().lower()
        else:
            var = var_spec
            requested_val = None
        
        # Try to match variable name case-insensitively
        matched_var = None
        for node_name in self.bn.nodes:
            if node_name.lower() == var.lower():
                matched_var = node_name
                break
        
        if matched_var is None:
            print(f"Unknown variable: {var}")
            print(f"Valid variables: {', '.join(sorted(self.bn.nodes.keys()))}")
            return
        
        var = matched_var  # Use the correctly-cased name
        
        dist = self.engine.query(var, self.evidence)
        
        if requested_val:
            # User asked for specific value
            if requested_val in dist:
                print(f"\nP({var}={requested_val} | evidence) = {dist[requested_val]:.6f}")
            else:
                print(f"Invalid value for {var}: {requested_val}")
                print(f"Valid values: {list(dist.keys())}")
        else:
            # Show full distribution
            print(f"\nP({var} | evidence):")
            for val, prob in sorted(dist.items()):
                print(f"  P({var}={val}) = {prob:.6f}")
    
    def query_path(self, spec: str):
        """
        Query probability that a path is clear.
        
        Format: comma-separated edge IDs (e.g., '1,2,3')
        """
        try:
            edge_ids = [int(e.strip()) for e in spec.split(',')]
            
            # Validate edges
            for eid in edge_ids:
                if eid not in self.bn.graph.edges:
                    print(f"Unknown edge: {eid}")
                    print(f"Valid edges: {list(self.bn.graph.edges.keys())}")
                    return
            
            prob = self.engine.query_path_clear(edge_ids, self.evidence)
            edge_str = ', '.join(f"E{e}" for e in edge_ids)
            print(f"\nP(path [{edge_str}] is clear | evidence) = {prob:.6f}")
        
        except ValueError as e:
            print(f"Usage: path <edge1>,<edge2>,...")
            print("Example: path 1,2,3")
    
    def find_safest_path(self):
        """
        Find the safest path between two vertices (bonus feature).
        
        This finds the path with highest probability of being clear.
        """
        try:
            start = int(input("Start vertex: "))
            goal = int(input("Goal vertex: "))
            
            if start not in self.bn.graph.get_vertices():
                print(f"Invalid start vertex: {start}")
                return
            if goal not in self.bn.graph.get_vertices():
                print(f"Invalid goal vertex: {goal}")
                return
            
            # Find all simple paths and evaluate their safety
            paths = self._find_all_paths(start, goal)
            
            if not paths:
                print(f"No path exists from {start} to {goal}")
                return
            
            print(f"\nFound {len(paths)} path(s) from {start} to {goal}:")
            
            best_prob = -1
            best_path = None
            
            for path_edges in paths:
                prob = self.engine.query_path_clear(path_edges, self.evidence)
                edge_str = ' -> '.join(f"E{e}" for e in path_edges)
                print(f"  Path [{edge_str}]: P(clear) = {prob:.6f}")
                
                if prob > best_prob:
                    best_prob = prob
                    best_path = path_edges
            
            if best_path:
                edge_str = ' -> '.join(f"E{e}" for e in best_path)
                print(f"\nSafest path: [{edge_str}] with P(clear) = {best_prob:.6f}")
        
        except ValueError:
            print("Please enter valid vertex numbers.")
    
    def _find_all_paths(self, start: int, goal: int, 
                        max_length: int = 10) -> list:
        """
        Find all simple paths from start to goal.
        
        Returns list of paths, where each path is a list of edge IDs.
        """
        paths = []
        
        # DFS to find all paths
        def dfs(current: int, target: int, visited: set, path_edges: list):
            if current == target:
                if path_edges:  # Don't return empty path
                    paths.append(list(path_edges))
                return
            
            if len(path_edges) >= max_length:
                return
            
            for eid in self.bn.graph.vertex_edges.get(current, []):
                edge = self.bn.graph.edges[eid]
                # Get the other endpoint
                neighbor = edge.v if edge.u == current else edge.u
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    path_edges.append(eid)
                    dfs(neighbor, target, visited, path_edges)
                    path_edges.pop()
                    visited.remove(neighbor)
        
        dfs(start, goal, {start}, [])
        return paths


def main():
    """Main entry point."""
    print("=" * 60)
    print("HURRICANE EVACUATION - BAYESIAN NETWORK (Assignment 3)")
    print("=" * 60)
    
    # Get input file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter input file path [test_input.txt]: ").strip()
        if not file_path:
            file_path = "test_input.txt"
    
    try:
        # Parse the input file
        print(f"\nLoading graph from: {file_path}")
        graph_data = parse_file(file_path)
        print("Graph loaded successfully!")
        print_graph_data(graph_data)
        
        # Build Bayesian Network
        print("\nConstructing Bayesian Network...")
        bn = BayesianNetwork(graph_data)
        bn.print_network_structure()
        bn.print_cpts()
        
        # Create inference engine
        engine = InferenceEngine(bn)
        
        # Start interactive session
        session = InteractiveSession(bn, engine)
        session.run()
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()