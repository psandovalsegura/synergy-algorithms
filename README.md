# synergy-algorithms

<!-- [![arXiv](http://img.shields.io/badge/arXiv- -B31B1B.svg)](http://arxiv.org/abs/ ) -->

Code for the paper "Heterogeneous Team Formation with Multiplicity using Synergy Graphs" by Sandoval-Segura et al. Below we outline which algorithms are from previous work.

A Python implementation of [Modeling and Learning Synergy for Team Formation with Heterogeneous Agents (2012)](https://dl.acm.org/doi/10.5555/2343576.2343628) by Liemhetcharat and Veloso, which includes algorithms for
- Approximating the optimal team of size n (Algorithm 1)
- Creating a Synergy Graph from Observations (Algorithm 2)
- Estimating the Individual Agent Capabilities (Algorithm 3) 

This repository also contains an implementation of the follow-up work [Weighted synergy graphs for role assignment in ad hoc heterogeneous robot teams (2012)](https://ieeexplore.ieee.org/document/6386027), which includes algorithms for
- Learning a Weighted Synergy Graph from Training Examples (Algorithm 1)

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

### Experimental code
Code used for designing and running experiments can be found in the [experiment-code](https://github.com/psandovalsegura/synergy-algorithms/tree/experiment-code) branch. 

<!-- More information can be found in our paper which is available on arXiv: https://arxiv.org/abs/. 

### Citation
```
```
-->