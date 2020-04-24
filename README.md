# synergy-algorithms
A Python implementation of [Modeling and Learning Synergy for Team Formation with Heterogeneous Agents (2012)](https://dl.acm.org/doi/10.5555/2343576.2343628) by Liemhetcharat and Veloso, which describes algorithms for:
- Approximating the optimal team of size n (Algorithm 1)
- Creating a Synergy Graph from Observations (Algorithm 2)
- Estimating the Individual Agent Capabilities (Algorithm 3) 

This repository also contains an implementation of the follow-up work [Weighted synergy graphs for role assignment in ad hoc heterogeneous robot teams (2012)](https://ieeexplore.ieee.org/document/6386027), which describes algorithms for:
- Learning a Weighted Synergy Graph from Training Examples (Algorithm 1)

Code by Pedro Sandoval Segura

### Requirements
All required modules can be installed with
```
pip install -r requirements.txt
```

But it is recommended you first create a virtual environment, then install modules with
```
python3 -m venv synergy-algorithms-venv;
source synergy-algorithms-venv/bin/activate;
pip install -r requirements.txt
```

### Tests
All test cases can be run from the root of the repository with
```
python -m pytest -vv tests/
```
