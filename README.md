# Assignment 3: Bayesian Network for Hurricane Evacuation

## Overview

This project implements a Bayesian Network for reasoning under uncertainty in the 
Hurricane Evacuation Problem. The network models relationships between weather 
conditions, road flooding, and the presence of evacuees.

## Files

- `parser.py` - Parses input files defining the graph and probability parameters
- `bayes_network.py` - Defines the Bayesian Network structure and CPTs
- `inference.py` - Implements Variable Elimination for probabilistic inference
- `main.py` - Interactive main program
- `test_input.txt` - Example scenario 1 (4 vertices)
- `scenario2.txt` - Example scenario 2 (5 vertices)

## Running the Program

```bash
python main.py [input_file]
```

If no input file is provided, you will be prompted to enter one.

### Interactive Commands

Once the program is running:

- `help` - Show available commands
- `add F1=true` - Add evidence that edge 1 is flooded
- `add W=stormy` - Add evidence that weather is stormy
- `add Ev2=true` - Add evidence that there are evacuees at vertex 2
- `reset` - Clear all evidence
- `all` - Show all posterior probabilities
- `prob F1` - Query P(F1 | evidence)
- `path 1,2,3` - Query probability that edges 1, 2, 3 are all clear
- `best` - Find safest path between two vertices (bonus)
- `quit` - Exit

## Input File Format

```
#V 4          ; number of vertices (1 to n)
#P1 0.3       ; noisy-OR parameter

#E1 1 3 W1 F 0.2  ; Edge 1: vertices 1-3, weight 1, P(flood|mild)=0.2
#E2 2 3 W3 F 0.1  ; Edge 2: vertices 2-3, weight 3, P(flood|mild)=0.1

#W 0.1 0.4 0.5    ; Prior: P(mild)=0.1, P(stormy)=0.4, P(extreme)=0.5
```

## Bayesian Network Structure

### Nodes

1. **Weather (W)**: Root node
   - Domain: {mild, stormy, extreme}
   - Prior probability from input file

2. **Flooding F(e)**: One per edge
   - Domain: {true, false}
   - Parents: Weather
   - P(F|mild) = base probability
   - P(F|stormy) = 2 × base (capped at 1.0)
   - P(F|extreme) = 3 × base (capped at 1.0)

3. **Evacuees Ev(v)**: One per vertex
   - Domain: {true, false}
   - Parents: All flooding nodes of incident edges
   - Uses Noisy-OR model

### Noisy-OR Model

The Noisy-OR model compactly represents how multiple causes (flooded edges) can 
independently lead to an effect (evacuees present).

For each flooded edge e incident to vertex v:
- q_e = min(1, P1 / weight(e))

Then:
- P(Ev=false | flooding states) = ∏(1 - q_e) for all flooded edges
- P(Ev=true | flooding states) = 1 - P(Ev=false)

**Key property**: If no incident edges are flooded, P(Ev=true) = 0.

## Inference Algorithm: Variable Elimination

The inference engine uses Variable Elimination, an exact algorithm for computing 
P(Query | Evidence).

### Algorithm Steps

1. **Initialize factors** from CPTs (one factor per node)
2. **Incorporate evidence** by restricting factors
3. **Eliminate hidden variables** one at a time:
   - Multiply all factors containing the variable
   - Sum out the variable
4. **Multiply remaining factors**
5. **Normalize** to get probabilities

### Why Variable Elimination?

- Exact inference (no approximation)
- More efficient than naive enumeration
- Exploits conditional independence structure
- Complexity depends on factor sizes, not total variables

## Example Scenarios

### Scenario 1 (test_input.txt)
A 4-vertex graph with edges of varying flood probabilities.

### Scenario 2 (scenario2.txt)
A 5-vertex graph demonstrating path finding between distant vertices.

## Sample Output

```
P(Weather=extreme | F1=true) = 0.625
P(F2=true | F1=true) = 0.258
P(Ev1=true | F1=true) = 0.300
```

Observing that edge 1 is flooded:
- Increases P(extreme weather) from 0.5 to 0.625
- Increases P(other edges flooded) due to shared weather cause
- P(Ev1=true) becomes exactly the noisy-OR parameter q1 = 0.3
