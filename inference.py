from typing import Dict, List, Tuple, Set, Optional, FrozenSet
from dataclasses import dataclass
from bayes_network import BayesianNetwork
import itertools
from collections import defaultdict


@dataclass(frozen=True)
class Factor:
    variables: Tuple[str, ...]
    table: Dict[Tuple[str, ...], float]
    
    def __repr__(self):
        return f"Factor({self.variables})"


class InferenceEngine:

    def __init__(self, bn: BayesianNetwork):
        self.bn = bn
    
    # =====================================================
    # FACTOR CONSTRUCTION
    # =====================================================
    
    def make_factor_weather(self) -> Factor:
        variables = ('W',)
        table = {}
        
        for w in ['mild', 'stormy', 'extreme']:
            table[(w,)] = self.bn.get_weather_prob(w)
        
        return Factor(variables, table)
    
    def make_factor_flooding(self, edge_id: int) -> Factor:
        flood_var = f"F{edge_id}"
        variables = ('W', flood_var)
        table = {}
        
        for w in ['mild', 'stormy', 'extreme']:
            for f in ['true', 'false']:
                flooded = (f == 'true')
                prob = self.bn.get_flooding_prob(edge_id, flooded, w)
                table[(w, f)] = prob
        
        return Factor(variables, table)
    
    def make_factor_evacuees(self, vertex: int) -> Factor:
        incident_edge_ids = self.bn.graph.vertex_edges.get(vertex, [])
        evac_var = f"Ev{vertex}"
        
        # Variables: all incident flooding variables + evacuees variable
        flood_vars = [f"F{eid}" for eid in incident_edge_ids]
        variables = tuple(flood_vars + [evac_var])
        
        table = {}
        
        if not incident_edge_ids:
            # No incident edges - evacuees always false
            table[('false',)] = 1.0
            table[('true',)] = 0.0
            return Factor((evac_var,), table)
        
        # Generate all combinations of flooding states
        n_edges = len(incident_edge_ids)
        for flood_combo in itertools.product(['false', 'true'], repeat=n_edges):
            # Build flooding state dictionary
            flooded_state = {
                incident_edge_ids[i]: (flood_combo[i] == 'true')
                for i in range(n_edges)
            }
            
            for ev in ['false', 'true']:
                has_evac = (ev == 'true')
                prob = self.bn.get_evacuees_prob(vertex, has_evac, flooded_state)
                assignment = flood_combo + (ev,)
                table[assignment] = prob
        
        return Factor(variables, table)
    
    def get_all_factors(self) -> List[Factor]:
        factors = []
        
        # Weather prior
        factors.append(self.make_factor_weather())
        
        # Flooding factors (one per edge)
        for edge_id in self.bn.graph.edges:
            factors.append(self.make_factor_flooding(edge_id))
        
        # Evacuees factors (one per vertex)
        for v in self.bn.graph.get_vertices():
            factors.append(self.make_factor_evacuees(v))
        
        return factors
    
    # =====================================================
    # FACTOR OPERATIONS
    # =====================================================
    
    def restrict_factor(self, factor: Factor, variable: str, value: str) -> Factor:
        if variable not in factor.variables:
            return factor  # Variable not in this factor
        
        var_idx = factor.variables.index(variable)
        new_vars = tuple(v for v in factor.variables if v != variable)
        new_table = {}
        
        for assignment, prob in factor.table.items():
            if assignment[var_idx] == value:
                # Remove the restricted variable from assignment
                new_assignment = tuple(
                    a for i, a in enumerate(assignment) if i != var_idx
                )
                if new_assignment:
                    new_table[new_assignment] = prob
                else:
                    # Factor becomes a constant
                    new_table[()] = prob
        
        return Factor(new_vars, new_table)
    
    def multiply_factors(self, factor1: Factor, factor2: Factor) -> Factor:
        # Find union of variables (preserving order)
        vars1 = list(factor1.variables)
        vars2 = list(factor2.variables)
        
        all_vars = vars1.copy()
        for v in vars2:
            if v not in all_vars:
                all_vars.append(v)
        
        new_vars = tuple(all_vars)
        new_table = {}
        
        # Get domains for all variables
        domains = self._get_domains(all_vars)
        
        # Iterate over all assignments to combined variables
        for assignment in itertools.product(*[domains[v] for v in all_vars]):
            assign_dict = dict(zip(all_vars, assignment))
            
            # Extract assignments for each factor
            assign1 = tuple(assign_dict[v] for v in factor1.variables) if factor1.variables else ()
            assign2 = tuple(assign_dict[v] for v in factor2.variables) if factor2.variables else ()
            
            # Get values from both factors
            val1 = factor1.table.get(assign1, 0.0)
            val2 = factor2.table.get(assign2, 0.0)
            
            new_table[assignment] = val1 * val2
        
        return Factor(new_vars, new_table)
    
    def sum_out(self, factor: Factor, variable: str) -> Factor:
        if variable not in factor.variables:
            return factor
        
        var_idx = factor.variables.index(variable)
        new_vars = tuple(v for v in factor.variables if v != variable)
        new_table = defaultdict(float)
        
        for assignment, prob in factor.table.items():
            # Create new assignment without the summed-out variable
            new_assignment = tuple(
                a for i, a in enumerate(assignment) if i != var_idx
            )
            if not new_assignment:
                new_assignment = ()
            new_table[new_assignment] += prob
        
        return Factor(new_vars, dict(new_table))
    
    def _get_domains(self, variables: List[str]) -> Dict[str, List[str]]:
        domains = {}
        for v in variables:
            if v in self.bn.nodes:
                domains[v] = self.bn.nodes[v].domain
            elif v == 'W':
                domains[v] = ['mild', 'stormy', 'extreme']
            else:
                domains[v] = ['true', 'false']
        return domains
    
    def normalize_factor(self, factor: Factor) -> Factor:
        total = sum(factor.table.values())
        if total == 0:
            return factor
        
        new_table = {k: v / total for k, v in factor.table.items()}
        return Factor(factor.variables, new_table)
    
    # =====================================================
    # VARIABLE ELIMINATION
    # =====================================================
    
    def variable_elimination(self, query_vars: List[str], 
                            evidence: Dict[str, str]) -> Factor:
        # Step 1: Get all factors
        factors = self.get_all_factors()
        
        # Step 2: Restrict factors by evidence
        for var, val in evidence.items():
            factors = [self.restrict_factor(f, var, val) for f in factors]
        
        # Step 3: Determine hidden variables (not query, not evidence)
        all_vars = set()
        for f in factors:
            all_vars.update(f.variables)
        
        query_set = set(query_vars)
        evidence_set = set(evidence.keys())
        hidden_vars = all_vars - query_set - evidence_set
        
        # Step 4: Eliminate hidden variables one by one
        elimination_order = self._get_elimination_order(hidden_vars, factors)
        
        for var in elimination_order:
            # Find all factors containing this variable
            relevant = [f for f in factors if var in f.variables]
            irrelevant = [f for f in factors if var not in f.variables]
            
            if relevant:
                # Multiply all relevant factors
                product = relevant[0]
                for f in relevant[1:]:
                    product = self.multiply_factors(product, f)
                
                # Sum out the variable
                new_factor = self.sum_out(product, var)
                
                # Update factor list
                factors = irrelevant + [new_factor]
        
        # Step 5: Multiply remaining factors
        if not factors:
            return Factor(tuple(query_vars), {(): 1.0})
        
        result = factors[0]
        for f in factors[1:]:
            result = self.multiply_factors(result, f)
        
        # Step 6: Normalize
        result = self.normalize_factor(result)
        
        return result
    
    def _get_elimination_order(self, hidden_vars: Set[str], 
                               factors: List[Factor]) -> List[str]:
        var_count = defaultdict(int)
        for var in hidden_vars:
            for f in factors:
                if var in f.variables:
                    var_count[var] += 1
        
        # Sort by count (ascending)
        return sorted(hidden_vars, key=lambda v: var_count[v])
    
    # =====================================================
    # QUERY METHODS
    # =====================================================
    
    def query(self, query_var: str, evidence: Dict[str, str] = None) -> Dict[str, float]:
        if evidence is None:
            evidence = {}
        
        # Special case: if query_var is in evidence, return deterministic
        if query_var in evidence:
            domain = self.bn.nodes[query_var].domain if query_var in self.bn.nodes else ['true', 'false']
            distribution = {val: 0.0 for val in domain}
            distribution[evidence[query_var]] = 1.0
            return distribution
        
        result = self.variable_elimination([query_var], evidence)
        
        # Convert factor to dictionary
        distribution = {}
        for assignment, prob in result.table.items():
            if len(assignment) == 1:
                distribution[assignment[0]] = prob
            elif len(assignment) == 0:
                # Empty assignment - shouldn't happen normally
                continue
            else:
                # Find index of query_var
                if query_var in result.variables:
                    idx = result.variables.index(query_var)
                    val = assignment[idx]
                    distribution[val] = distribution.get(val, 0) + prob
        
        return distribution
    
    def query_path_clear(self, edge_ids: List[int], 
                         evidence: Dict[str, str] = None) -> float:
        if evidence is None:
            evidence = {}
        
        # Create a copy of evidence
        extended_evidence = dict(evidence)
        
        flood_vars = [f"F{eid}" for eid in edge_ids]
        
        result = self.variable_elimination(flood_vars, evidence)
        
        # Find the assignment where all are 'false'
        all_clear = tuple('false' for _ in edge_ids)
        
        # We need to extract just this probability
        prob_clear = result.table.get(all_clear, 0.0)
        
        return prob_clear
    
    def print_all_posteriors(self, evidence: Dict[str, str] = None):
        if evidence is None:
            evidence = {}
        
        print("\n" + "=" * 60)
        print("POSTERIOR PROBABILITIES")
        if evidence:
            print(f"Evidence: {evidence}")
        else:
            print("Evidence: None (prior probabilities)")
        print("=" * 60)
        
        # 1. Weather distribution
        print("\nWeather Distribution:")
        weather_dist = self.query('W', evidence)
        for w in ['mild', 'stormy', 'extreme']:
            print(f"  P(Weather={w} | evidence) = {weather_dist.get(w, 0):.6f}")
        
        # 2. Flooding probabilities
        print("\nFlooding Probabilities:")
        for eid in sorted(self.bn.graph.edges.keys()):
            flood_dist = self.query(f'F{eid}', evidence)
            print(f"  P(F{eid}=true | evidence) = {flood_dist.get('true', 0):.6f}")
        
        # 3. Evacuees probabilities  
        print("\nEvacuees Probabilities:")
        for v in self.bn.graph.get_vertices():
            evac_dist = self.query(f'Ev{v}', evidence)
            print(f"  P(Ev{v}=true | evidence) = {evac_dist.get('true', 0):.6f}")


# Test code
if __name__ == "__main__":
    from parser import parse_file
    from bayes_network import BayesianNetwork
    
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
    
    with open("test_inference.txt", "w") as f:
        f.write(test_input)
    
    # Parse and build network
    data = parse_file("test_inference.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    # Test queries
    print("Testing Inference Engine")
    print("========================")
    
    # Prior probabilities (no evidence)
    engine.print_all_posteriors()
    
    # With some evidence
    print("\n\nWith evidence: Edge 1 is flooded")
    engine.print_all_posteriors({'F1': 'true'})
    
    # Path query
    print("\n\nPath Query: P(E1 and E2 clear) = ", 
          engine.query_path_clear([1, 2]))
    print("Path Query with F1=true evidence: P(E2 and E3 clear | F1=true) = ",
          engine.query_path_clear([2, 3], {'F1': 'true'}))
