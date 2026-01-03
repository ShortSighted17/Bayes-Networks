"""
Test Suite for Assignment 3: Bayesian Network for Hurricane Evacuation

This test suite verifies:
1. Parser correctness
2. Bayesian Network structure
3. Conditional Probability Tables (CPTs)
4. Inference engine correctness
5. Edge cases and special scenarios

Run with: python tests.py
"""

import math
import sys
from typing import Dict

# Import our modules
from parser import parse_file, GraphData, Edge
from bayes_network import BayesianNetwork
from inference import InferenceEngine, Factor


# ============================================================
# TEST UTILITIES
# ============================================================

def approx_equal(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) < tolerance


def assert_approx(actual: float, expected: float, message: str, tolerance: float = 1e-6):
    """Assert that actual ≈ expected, with a helpful message on failure."""
    if not approx_equal(actual, expected, tolerance):
        print(f"  FAILED: {message}")
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")
        return False
    return True


def assert_true(condition: bool, message: str):
    """Assert that condition is true."""
    if not condition:
        print(f"  FAILED: {message}")
        return False
    return True


def run_test(name: str, test_func):
    """Run a test function and report results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    try:
        passed = test_func()
        if passed:
            print(f"  ✓ PASSED")
        return passed
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# TEST INPUT FILES
# ============================================================

# Simple 2-vertex graph for basic testing
SIMPLE_INPUT = """
#V 2
#P1 0.5

#E1 1 2 W1 F 0.3

#W 0.5 0.3 0.2
"""

# The example from the assignment
ASSIGNMENT_INPUT = """
#V 4
#P1 0.3

#E1 1 3 W1 F 0.2
#E2 2 3 W3 F 0.1
#E3 2 4 W3 F 0.3
#E4 3 4 W4 F 0

#W 0.1 0.4 0.5
"""

# Edge case: vertex with no edges
ISOLATED_VERTEX_INPUT = """
#V 3
#P1 0.4

#E1 1 2 W1 F 0.5

#W 0.33 0.33 0.34
"""


def create_test_file(content: str, filename: str):
    """Write test content to a file."""
    with open(filename, 'w') as f:
        f.write(content)


# ============================================================
# PARSER TESTS
# ============================================================

def test_parser_basic():
    """Test that parser correctly reads vertices, edges, and parameters."""
    create_test_file(ASSIGNMENT_INPUT, "test_parser.txt")
    data = parse_file("test_parser.txt")
    
    all_passed = True
    
    # Check number of vertices
    all_passed &= assert_true(data.num_vertices == 4, "Should have 4 vertices")
    
    # Check P1 parameter
    all_passed &= assert_approx(data.p1, 0.3, "P1 should be 0.3")
    
    # Check weather prior
    all_passed &= assert_approx(data.weather_prior[0], 0.1, "P(mild) should be 0.1")
    all_passed &= assert_approx(data.weather_prior[1], 0.4, "P(stormy) should be 0.4")
    all_passed &= assert_approx(data.weather_prior[2], 0.5, "P(extreme) should be 0.5")
    
    # Check edges
    all_passed &= assert_true(len(data.edges) == 4, "Should have 4 edges")
    all_passed &= assert_true(data.edges[1].u == 1 and data.edges[1].v == 3, "Edge 1 connects 1-3")
    all_passed &= assert_approx(data.edges[1].flood_prob_mild, 0.2, "Edge 1 flood prob should be 0.2")
    all_passed &= assert_true(data.edges[1].weight == 1, "Edge 1 weight should be 1")
    
    # Check vertex-edge mapping
    all_passed &= assert_true(1 in data.vertex_edges[1], "Vertex 1 should be incident to edge 1")
    all_passed &= assert_true(1 in data.vertex_edges[3], "Vertex 3 should be incident to edge 1")
    
    return all_passed


def test_parser_weather_scaling():
    """Test that flood probabilities scale correctly with weather."""
    create_test_file(SIMPLE_INPUT, "test_weather.txt")
    data = parse_file("test_weather.txt")
    
    edge = data.edges[1]
    all_passed = True
    
    # P(flood|mild) = 0.3
    all_passed &= assert_approx(edge.flood_prob_given_weather('mild'), 0.3, 
                                "P(flood|mild) should be 0.3")
    
    # P(flood|stormy) = 2 * 0.3 = 0.6
    all_passed &= assert_approx(edge.flood_prob_given_weather('stormy'), 0.6, 
                                "P(flood|stormy) should be 0.6")
    
    # P(flood|extreme) = 3 * 0.3 = 0.9
    all_passed &= assert_approx(edge.flood_prob_given_weather('extreme'), 0.9, 
                                "P(flood|extreme) should be 0.9")
    
    return all_passed


def test_parser_flood_prob_capped():
    """Test that flood probability is capped at 1.0."""
    # Create input with high base flood probability
    high_flood_input = """
#V 2
#P1 0.5
#E1 1 2 W1 F 0.5
#W 0.33 0.33 0.34
"""
    create_test_file(high_flood_input, "test_cap.txt")
    data = parse_file("test_cap.txt")
    
    edge = data.edges[1]
    all_passed = True
    
    # P(flood|stormy) = 2 * 0.5 = 1.0 (capped)
    all_passed &= assert_approx(edge.flood_prob_given_weather('stormy'), 1.0, 
                                "P(flood|stormy) should be capped at 1.0")
    
    # P(flood|extreme) = 3 * 0.5 = 1.5 -> capped to 1.0
    all_passed &= assert_approx(edge.flood_prob_given_weather('extreme'), 1.0, 
                                "P(flood|extreme) should be capped at 1.0")
    
    return all_passed


# ============================================================
# BAYESIAN NETWORK STRUCTURE TESTS
# ============================================================

def test_bn_structure():
    """Test that BN has correct nodes and parent-child relationships."""
    create_test_file(ASSIGNMENT_INPUT, "test_bn.txt")
    data = parse_file("test_bn.txt")
    bn = BayesianNetwork(data)
    
    all_passed = True
    
    # Should have 1 weather + 4 flooding + 4 evacuees = 9 nodes
    all_passed &= assert_true(len(bn.nodes) == 9, "Should have 9 nodes total")
    
    # Weather node
    all_passed &= assert_true('W' in bn.nodes, "Should have Weather node")
    all_passed &= assert_true(bn.nodes['W'].parents == [], "Weather should have no parents")
    all_passed &= assert_true(len(bn.nodes['W'].children) == 4, "Weather should have 4 children (flooding nodes)")
    
    # Flooding nodes
    for i in range(1, 5):
        fname = f'F{i}'
        all_passed &= assert_true(fname in bn.nodes, f"Should have {fname} node")
        all_passed &= assert_true(bn.nodes[fname].parents == ['W'], f"{fname} should have W as parent")
    
    # Evacuees nodes - check parents are correct flooding nodes
    # Vertex 1 is incident to edge 1 only
    all_passed &= assert_true(bn.nodes['Ev1'].parents == ['F1'], 
                              "Ev1 should have F1 as parent")
    # Vertex 3 is incident to edges 1, 2, 4
    all_passed &= assert_true(set(bn.nodes['Ev3'].parents) == {'F1', 'F2', 'F4'}, 
                              "Ev3 should have F1, F2, F4 as parents")
    
    return all_passed


# ============================================================
# CONDITIONAL PROBABILITY TABLE TESTS
# ============================================================

def test_cpt_weather():
    """Test weather prior probabilities."""
    create_test_file(ASSIGNMENT_INPUT, "test_cpt.txt")
    data = parse_file("test_cpt.txt")
    bn = BayesianNetwork(data)
    
    all_passed = True
    
    all_passed &= assert_approx(bn.get_weather_prob('mild'), 0.1, "P(mild) should be 0.1")
    all_passed &= assert_approx(bn.get_weather_prob('stormy'), 0.4, "P(stormy) should be 0.4")
    all_passed &= assert_approx(bn.get_weather_prob('extreme'), 0.5, "P(extreme) should be 0.5")
    
    # Probabilities should sum to 1
    total = bn.get_weather_prob('mild') + bn.get_weather_prob('stormy') + bn.get_weather_prob('extreme')
    all_passed &= assert_approx(total, 1.0, "Weather probabilities should sum to 1.0")
    
    return all_passed


def test_cpt_flooding():
    """Test flooding conditional probabilities."""
    create_test_file(ASSIGNMENT_INPUT, "test_cpt2.txt")
    data = parse_file("test_cpt2.txt")
    bn = BayesianNetwork(data)
    
    all_passed = True
    
    # Edge 1: base prob 0.2
    all_passed &= assert_approx(bn.get_flooding_prob(1, True, 'mild'), 0.2, 
                                "P(F1=true|mild) should be 0.2")
    all_passed &= assert_approx(bn.get_flooding_prob(1, True, 'stormy'), 0.4, 
                                "P(F1=true|stormy) should be 0.4")
    all_passed &= assert_approx(bn.get_flooding_prob(1, True, 'extreme'), 0.6, 
                                "P(F1=true|extreme) should be 0.6")
    
    # P(flooded) + P(not flooded) = 1
    all_passed &= assert_approx(
        bn.get_flooding_prob(1, True, 'mild') + bn.get_flooding_prob(1, False, 'mild'), 
        1.0, "P(F1=true|mild) + P(F1=false|mild) should be 1.0")
    
    return all_passed


def test_cpt_evacuees_noisy_or():
    """Test evacuees noisy-OR conditional probabilities."""
    create_test_file(ASSIGNMENT_INPUT, "test_noisy_or.txt")
    data = parse_file("test_noisy_or.txt")
    bn = BayesianNetwork(data)
    
    all_passed = True
    
    # Vertex 1: only edge 1 (weight 1), q = min(1, 0.3/1) = 0.3
    # If no flooding: P(Ev) = 0
    all_passed &= assert_approx(
        bn.get_evacuees_prob(1, True, {1: False}), 0.0,
        "P(Ev1=true | F1=false) should be 0.0")
    
    # If flooded: P(Ev) = 1 - (1-0.3) = 0.3
    all_passed &= assert_approx(
        bn.get_evacuees_prob(1, True, {1: True}), 0.3,
        "P(Ev1=true | F1=true) should be 0.3")
    
    # Vertex 2: edges 2 (weight 3) and 3 (weight 3)
    # q2 = min(1, 0.3/3) = 0.1
    # q3 = min(1, 0.3/3) = 0.1
    
    # Both flooded: P(Ev) = 1 - (1-0.1)(1-0.1) = 1 - 0.81 = 0.19
    all_passed &= assert_approx(
        bn.get_evacuees_prob(2, True, {2: True, 3: True}), 0.19,
        "P(Ev2=true | F2=true, F3=true) should be 0.19")
    
    # Only one flooded: P(Ev) = 1 - (1-0.1) = 0.1
    all_passed &= assert_approx(
        bn.get_evacuees_prob(2, True, {2: True, 3: False}), 0.1,
        "P(Ev2=true | F2=true, F3=false) should be 0.1")
    
    return all_passed


# ============================================================
# INFERENCE ENGINE TESTS
# ============================================================

def test_inference_prior():
    """Test that inference without evidence returns prior probabilities."""
    create_test_file(SIMPLE_INPUT, "test_prior.txt")
    data = parse_file("test_prior.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # Query weather with no evidence - should return prior
    weather_dist = engine.query('W', {})
    all_passed &= assert_approx(weather_dist['mild'], 0.5, "P(mild) should be 0.5")
    all_passed &= assert_approx(weather_dist['stormy'], 0.3, "P(stormy) should be 0.3")
    all_passed &= assert_approx(weather_dist['extreme'], 0.2, "P(extreme) should be 0.2")
    
    return all_passed


def test_inference_evidence_on_query():
    """Test querying a variable that is in evidence."""
    create_test_file(SIMPLE_INPUT, "test_ev_query.txt")
    data = parse_file("test_ev_query.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # If we observe W=stormy, P(W=stormy|evidence) = 1.0
    weather_dist = engine.query('W', {'W': 'stormy'})
    all_passed &= assert_approx(weather_dist['stormy'], 1.0, 
                                "P(stormy|W=stormy) should be 1.0")
    all_passed &= assert_approx(weather_dist['mild'], 0.0, 
                                "P(mild|W=stormy) should be 0.0")
    
    return all_passed


def test_inference_predictive():
    """Test predictive inference: P(effect | cause)."""
    create_test_file(SIMPLE_INPUT, "test_pred.txt")
    data = parse_file("test_pred.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # P(F1=true | W=extreme)
    # Edge 1 has base prob 0.3, so extreme = 3*0.3 = 0.9
    flood_dist = engine.query('F1', {'W': 'extreme'})
    all_passed &= assert_approx(flood_dist['true'], 0.9, 
                                "P(F1=true|W=extreme) should be 0.9")
    
    # P(F1=true | W=mild) should be 0.3
    flood_dist = engine.query('F1', {'W': 'mild'})
    all_passed &= assert_approx(flood_dist['true'], 0.3, 
                                "P(F1=true|W=mild) should be 0.3")
    
    return all_passed


def test_inference_diagnostic():
    """Test diagnostic inference: P(cause | effect) - Bayesian reasoning!"""
    create_test_file(SIMPLE_INPUT, "test_diag.txt")
    data = parse_file("test_diag.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # First, calculate P(F1=true) without evidence
    # P(F1) = P(F1|mild)P(mild) + P(F1|stormy)P(stormy) + P(F1|extreme)P(extreme)
    #       = 0.3*0.5 + 0.6*0.3 + 0.9*0.2 = 0.15 + 0.18 + 0.18 = 0.51
    flood_prior = engine.query('F1', {})
    all_passed &= assert_approx(flood_prior['true'], 0.51, 
                                "P(F1=true) should be 0.51")
    
    # Now, P(W=extreme | F1=true) using Bayes' theorem:
    # P(W=extreme|F1) = P(F1|extreme)P(extreme) / P(F1)
    #                 = (0.9 * 0.2) / 0.51 = 0.18 / 0.51 ≈ 0.353
    weather_given_flood = engine.query('W', {'F1': 'true'})
    expected = (0.9 * 0.2) / 0.51
    all_passed &= assert_approx(weather_given_flood['extreme'], expected, 
                                f"P(W=extreme|F1=true) should be {expected:.6f}")
    
    # The probability of extreme weather should INCREASE when we see flooding
    # (because flooding is more likely in extreme weather)
    all_passed &= assert_true(weather_given_flood['extreme'] > 0.2,
                              "P(extreme|F1=true) should be > P(extreme)=0.2")
    
    return all_passed


def test_inference_intercausal():
    """Test inter-causal inference: correlation through common cause."""
    create_test_file(ASSIGNMENT_INPUT, "test_intercausal.txt")
    data = parse_file("test_intercausal.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # P(F2=true) without evidence
    f2_prior = engine.query('F2', {})
    
    # P(F2=true | F1=true) - should be HIGHER than prior
    # Because F1 flooded suggests bad weather, which makes F2 more likely
    f2_given_f1 = engine.query('F2', {'F1': 'true'})
    
    all_passed &= assert_true(f2_given_f1['true'] > f2_prior['true'],
                              "P(F2=true|F1=true) should be > P(F2=true) due to common cause")
    
    print(f"  P(F2=true) = {f2_prior['true']:.6f}")
    print(f"  P(F2=true|F1=true) = {f2_given_f1['true']:.6f}")
    
    return all_passed


def test_inference_noisy_or_deterministic():
    """Test that noisy-OR gives deterministic result in special case."""
    create_test_file(ASSIGNMENT_INPUT, "test_det.txt")
    data = parse_file("test_det.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # If we observe Ev1=true (evacuees at vertex 1), and vertex 1 only 
    # touches edge 1, then F1 MUST be true (since noisy-OR has no leak)
    f1_given_ev1 = engine.query('F1', {'Ev1': 'true'})
    all_passed &= assert_approx(f1_given_ev1['true'], 1.0,
                                "P(F1=true|Ev1=true) should be 1.0 (deterministic)")
    
    # Conversely, if F1=false, then Ev1 must be false
    ev1_given_not_f1 = engine.query('Ev1', {'F1': 'false'})
    all_passed &= assert_approx(ev1_given_not_f1['true'], 0.0,
                                "P(Ev1=true|F1=false) should be 0.0")
    
    return all_passed


def test_inference_normalization():
    """Test that query results are properly normalized (sum to 1)."""
    create_test_file(ASSIGNMENT_INPUT, "test_norm.txt")
    data = parse_file("test_norm.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # Test with various evidence combinations
    test_cases = [
        {},
        {'F1': 'true'},
        {'W': 'stormy'},
        {'F1': 'true', 'F2': 'false'},
        {'Ev3': 'true'},
    ]
    
    for evidence in test_cases:
        # Weather should sum to 1
        w_dist = engine.query('W', evidence)
        w_sum = sum(w_dist.values())
        all_passed &= assert_approx(w_sum, 1.0, 
                                    f"Weather probs should sum to 1 with evidence {evidence}")
        
        # Flooding should sum to 1
        f1_dist = engine.query('F1', evidence)
        f1_sum = sum(f1_dist.values())
        all_passed &= assert_approx(f1_sum, 1.0,
                                    f"F1 probs should sum to 1 with evidence {evidence}")
    
    return all_passed


def test_inference_path_clear():
    """Test path clearance probability queries."""
    create_test_file(ASSIGNMENT_INPUT, "test_path.txt")
    data = parse_file("test_path.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # P(E4 clear) should be 1.0 since edge 4 has flood prob 0
    p_e4_clear = engine.query_path_clear([4], {})
    all_passed &= assert_approx(p_e4_clear, 1.0,
                                "P(E4 clear) should be 1.0 (flood prob is 0)")
    
    # P(E1 clear) = 1 - P(F1=true)
    f1_dist = engine.query('F1', {})
    expected_e1_clear = 1.0 - f1_dist['true']
    p_e1_clear = engine.query_path_clear([1], {})
    all_passed &= assert_approx(p_e1_clear, expected_e1_clear,
                                f"P(E1 clear) should be {expected_e1_clear:.6f}")
    
    # If F1=true is evidence, P(E1 clear | F1=true) = 0
    p_e1_clear_given_f1 = engine.query_path_clear([1], {'F1': 'true'})
    all_passed &= assert_approx(p_e1_clear_given_f1, 0.0,
                                "P(E1 clear | F1=true) should be 0.0")
    
    return all_passed


def test_inference_multiple_evidence():
    """Test inference with multiple pieces of evidence."""
    create_test_file(ASSIGNMENT_INPUT, "test_multi.txt")
    data = parse_file("test_multi.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # With F1=true and F2=true, weather should shift even more toward extreme
    w_given_f1 = engine.query('W', {'F1': 'true'})
    w_given_f1_f2 = engine.query('W', {'F1': 'true', 'F2': 'true'})
    
    all_passed &= assert_true(w_given_f1_f2['extreme'] >= w_given_f1['extreme'],
                              "More flooding evidence should increase P(extreme)")
    
    print(f"  P(extreme|F1=true) = {w_given_f1['extreme']:.6f}")
    print(f"  P(extreme|F1=true,F2=true) = {w_given_f1_f2['extreme']:.6f}")
    
    return all_passed


# ============================================================
# FACTOR OPERATION TESTS
# ============================================================

def test_factor_multiply():
    """Test factor multiplication."""
    create_test_file(SIMPLE_INPUT, "test_mult.txt")
    data = parse_file("test_mult.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # Use actual factors from the BN to test multiplication
    # Get the weather factor and a flooding factor
    weather_factor = engine.make_factor_weather()
    flood_factor = engine.make_factor_flooding(1)
    
    # Multiply them
    result = engine.multiply_factors(weather_factor, flood_factor)
    
    # Result should be over (W, F1)
    all_passed &= assert_true('W' in result.variables, "Result should contain W")
    all_passed &= assert_true('F1' in result.variables, "Result should contain F1")
    
    # Check that P(W=mild, F1=true) = P(W=mild) * P(F1=true|W=mild)
    # P(mild) = 0.5 (from SIMPLE_INPUT), P(F1=true|mild) = 0.3
    expected = 0.5 * 0.3  # = 0.15
    
    # Find the correct entry
    for assignment, prob in result.table.items():
        assign_dict = dict(zip(result.variables, assignment))
        if assign_dict.get('W') == 'mild' and assign_dict.get('F1') == 'true':
            all_passed &= assert_approx(prob, expected, 
                                       f"P(mild, F1=true) should be {expected}")
            break
    
    return all_passed


def test_factor_sum_out():
    """Test summing out a variable from a factor."""
    create_test_file(SIMPLE_INPUT, "test_sum.txt")
    data = parse_file("test_sum.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # Factor over (A, B)
    f = Factor(('A', 'B'), {
        ('a1', 'b1'): 0.1,
        ('a1', 'b2'): 0.2,
        ('a2', 'b1'): 0.3,
        ('a2', 'b2'): 0.4,
    })
    
    # Sum out B
    result = engine.sum_out(f, 'B')
    
    all_passed &= assert_true(result.variables == ('A',), "Result should only have A")
    all_passed &= assert_approx(result.table[('a1',)], 0.3, "P(a1) = 0.1 + 0.2 = 0.3")
    all_passed &= assert_approx(result.table[('a2',)], 0.7, "P(a2) = 0.3 + 0.4 = 0.7")
    
    return all_passed


def test_factor_restrict():
    """Test restricting a factor by evidence."""
    create_test_file(SIMPLE_INPUT, "test_restrict.txt")
    data = parse_file("test_restrict.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # Factor over (A, B)
    f = Factor(('A', 'B'), {
        ('a1', 'b1'): 0.1,
        ('a1', 'b2'): 0.2,
        ('a2', 'b1'): 0.3,
        ('a2', 'b2'): 0.4,
    })
    
    # Restrict B=b1
    result = engine.restrict_factor(f, 'B', 'b1')
    
    all_passed &= assert_true(result.variables == ('A',), "Result should only have A")
    all_passed &= assert_approx(result.table[('a1',)], 0.1, "P(a1|b1) = 0.1")
    all_passed &= assert_approx(result.table[('a2',)], 0.3, "P(a2|b1) = 0.3")
    
    return all_passed


# ============================================================
# EDGE CASE TESTS
# ============================================================

def test_edge_case_zero_flood_prob():
    """Test edge with zero flooding probability."""
    create_test_file(ASSIGNMENT_INPUT, "test_zero.txt")
    data = parse_file("test_zero.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # Edge 4 has flood prob 0 in all weather conditions
    f4_dist = engine.query('F4', {})
    all_passed &= assert_approx(f4_dist['true'], 0.0,
                                "P(F4=true) should be 0.0")
    all_passed &= assert_approx(f4_dist['false'], 1.0,
                                "P(F4=false) should be 1.0")
    
    # Even with extreme weather
    f4_dist_extreme = engine.query('F4', {'W': 'extreme'})
    all_passed &= assert_approx(f4_dist_extreme['true'], 0.0,
                                "P(F4=true|W=extreme) should be 0.0")
    
    return all_passed


def test_edge_case_certain_evidence():
    """Test that certain evidence propagates correctly."""
    create_test_file(ASSIGNMENT_INPUT, "test_certain.txt")
    data = parse_file("test_certain.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # If weather is certainly mild, flooding probs should match CPT exactly
    f1_given_mild = engine.query('F1', {'W': 'mild'})
    all_passed &= assert_approx(f1_given_mild['true'], 0.2,
                                "P(F1=true|W=mild) should be 0.2")
    
    f3_given_mild = engine.query('F3', {'W': 'mild'})
    all_passed &= assert_approx(f3_given_mild['true'], 0.3,
                                "P(F3=true|W=mild) should be 0.3")
    
    return all_passed


# ============================================================
# CONSISTENCY TESTS
# ============================================================

def test_consistency_bayes_rule():
    """Verify Bayes' rule: P(A|B)P(B) = P(B|A)P(A)."""
    create_test_file(SIMPLE_INPUT, "test_bayes.txt")
    data = parse_file("test_bayes.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # P(W=extreme|F1=true) * P(F1=true) should equal 
    # P(F1=true|W=extreme) * P(W=extreme)
    
    p_extreme = engine.query('W', {})['extreme']  # 0.2
    p_f1 = engine.query('F1', {})['true']  # 0.51
    p_extreme_given_f1 = engine.query('W', {'F1': 'true'})['extreme']
    p_f1_given_extreme = engine.query('F1', {'W': 'extreme'})['true']  # 0.9
    
    lhs = p_extreme_given_f1 * p_f1
    rhs = p_f1_given_extreme * p_extreme
    
    all_passed &= assert_approx(lhs, rhs, 
                                f"Bayes rule: {lhs:.6f} should equal {rhs:.6f}")
    
    print(f"  P(extreme|F1) * P(F1) = {p_extreme_given_f1:.4f} * {p_f1:.4f} = {lhs:.6f}")
    print(f"  P(F1|extreme) * P(extreme) = {p_f1_given_extreme:.4f} * {p_extreme:.4f} = {rhs:.6f}")
    
    return all_passed


def test_consistency_marginalization():
    """Verify that marginalizing gives consistent results."""
    create_test_file(SIMPLE_INPUT, "test_marg.txt")
    data = parse_file("test_marg.txt")
    bn = BayesianNetwork(data)
    engine = InferenceEngine(bn)
    
    all_passed = True
    
    # P(F1) should equal sum over weather of P(F1|W)P(W)
    p_f1_direct = engine.query('F1', {})['true']
    
    p_f1_manual = 0.0
    for w in ['mild', 'stormy', 'extreme']:
        p_w = engine.query('W', {})[w]
        p_f1_given_w = engine.query('F1', {'W': w})['true']
        p_f1_manual += p_f1_given_w * p_w
    
    all_passed &= assert_approx(p_f1_direct, p_f1_manual,
                                f"P(F1) direct={p_f1_direct:.6f} should equal marginalized={p_f1_manual:.6f}")
    
    return all_passed


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "=" * 60)
    print("BAYESIAN NETWORK TEST SUITE")
    print("=" * 60)
    
    tests = [
        # Parser tests
        ("Parser - Basic Reading", test_parser_basic),
        ("Parser - Weather Scaling", test_parser_weather_scaling),
        ("Parser - Flood Prob Capped", test_parser_flood_prob_capped),
        
        # BN structure tests
        ("BN Structure", test_bn_structure),
        
        # CPT tests
        ("CPT - Weather Prior", test_cpt_weather),
        ("CPT - Flooding", test_cpt_flooding),
        ("CPT - Noisy-OR Evacuees", test_cpt_evacuees_noisy_or),
        
        # Inference tests
        ("Inference - Prior (No Evidence)", test_inference_prior),
        ("Inference - Evidence on Query Variable", test_inference_evidence_on_query),
        ("Inference - Predictive (Cause→Effect)", test_inference_predictive),
        ("Inference - Diagnostic (Effect→Cause)", test_inference_diagnostic),
        ("Inference - Inter-causal Reasoning", test_inference_intercausal),
        ("Inference - Noisy-OR Deterministic Case", test_inference_noisy_or_deterministic),
        ("Inference - Normalization", test_inference_normalization),
        ("Inference - Path Clearance", test_inference_path_clear),
        ("Inference - Multiple Evidence", test_inference_multiple_evidence),
        
        # Factor operation tests
        ("Factor - Multiplication", test_factor_multiply),
        ("Factor - Sum Out", test_factor_sum_out),
        ("Factor - Restrict", test_factor_restrict),
        
        # Edge cases
        ("Edge Case - Zero Flood Probability", test_edge_case_zero_flood_prob),
        ("Edge Case - Certain Evidence", test_edge_case_certain_evidence),
        
        # Consistency tests
        ("Consistency - Bayes Rule", test_consistency_bayes_rule),
        ("Consistency - Marginalization", test_consistency_marginalization),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        if run_test(name, test_func):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")
    print("=" * 60)
    
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
