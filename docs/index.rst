pg-neo-graph-rl Documentation
===============================

Welcome to the documentation for **pg-neo-graph-rl**, a federated graph-neural reinforcement learning toolkit for city-scale infrastructure control.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/index
   api/index
   examples/index
   deployment/index

Overview
--------

**pg-neo-graph-rl** combines cutting-edge dynamic graph neural networks with federated reinforcement learning for distributed control of city-scale infrastructure. Building on new dynamic-graph methods and merging them with federated actor-critic loops, this toolkit enables scalable, privacy-preserving control of traffic networks, power grids, and autonomous swarms.

Key Features
------------

* **Gossip Parameter Server**: Decentralized learning without central coordination
* **JAX-Accelerated Backend**: Blazing fast graph operations and RL training  
* **Dynamic Graph Support**: Handles time-varying topologies and edge attributes
* **Grafana Integration**: Real-time monitoring of distributed learning metrics
* **Privacy-Preserving**: Federated learning keeps sensitive data local

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

    # Clone repository
    git clone https://github.com/terragon-labs/pg-neo-graph-rl.git
    cd pg-neo-graph-rl

    # Install with JAX GPU support
    pip install -e ".[gpu]"

    # For CPU-only installation
    pip install -e ".[cpu]"

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from pg_neo_graph_rl import FederatedGraphRL, TrafficEnvironment
    from pg_neo_graph_rl.algorithms import GraphPPO

    # Initialize environment
    env = TrafficEnvironment(city="manhattan", num_intersections=2456)
    
    # Create federated learning system
    fed_system = FederatedGraphRL(num_agents=100, aggregation="gossip")
    
    # Train agents
    agents = [GraphPPO(agent_id=i) for i in range(100)]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`