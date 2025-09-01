
### **Project 8: Network Topology Optimization**

#### **1. Objective**
To model a computer network as a graph and use graph analytics algorithms to identify critical nodes, discover community structures (network segments), and find the most efficient paths. This project provides a foundation in thinking about network problems from a structural perspective.

#### **2. Business Value**
Graph analytics allows us to optimize network design and performance in ways that are not obvious from looking at device-level metrics alone:
*   **Improved Resilience:** By identifying critical nodes (potential single points of failure), we can prioritize them for redundancy and hardening.
*   **Enhanced Security:** Community detection algorithms can help identify unexpected connections between network segments that may violate security policies.
*   **Efficient Routing:** Graph algorithms are the foundation of routing protocols (like OSPF) and can be used to calculate the most efficient paths based on various metrics (latency, cost, bandwidth).

#### **3. Core Libraries**
*   `networkx`: The standard Python library for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
*   `pandas`: To load and structure the data that defines the network.
*   `matplotlib`: For visualizing the network graph.

#### **4. Dataset**
*   **Primary Dataset:** **The Internet Topology Zoo** ([Verified Link to Website](http://www.topology-zoo.org/dataset.html))
*   **Why it's suitable:** The Topology Zoo is a collection of over 200 real-world network topologies from various network operators and research institutions. The data is available in `graphml` format, which can be loaded directly into `networkx`. This provides a rich and realistic dataset for practicing graph analytics.
*   **Action:** For this project, we will download the `GtsCe.graphml` file, which represents the Central European network of GEANT, the pan-European research and education network.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a project folder and a Python virtual environment.
    ```bash
    mkdir network-topology
    cd network-topology
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the necessary libraries. `networkx` is the key component.
    ```bash
    pip install pandas networkx matplotlib jupyterlab
    ```
3.  Start a Jupyter Lab session.
    ```bash
    jupyter lab
    ```

**Step 2: Load the Network Graph**
1.  Download the `GtsCe.graphml` file from the Topology Zoo website and place it in your project folder.
2.  In your Jupyter Notebook, use `networkx` to load this file directly into a graph object.
    ```python
    import networkx as nx
    import matplotlib.pyplot as plt

    # Load the graph from the .graphml file
    G = nx.read_graphml('GtsCe.graphml')

    # Print some basic information about the graph
    print(f"Number of nodes (routers/switches): {G.number_of_nodes()}")
    print(f"Number of edges (connections): {G.number_of_edges()}")

    # Display information about the first few nodes
    print("\nSample nodes:")
    for i, node_data in enumerate(G.nodes(data=True)):
        print(node_data)
        if i >= 4: break
    ```
    *You will see that the nodes have attributes like Country, City, and geographic coordinates.*

**Step 3: Visualize the Network Topology**
1.  A visual representation is the most intuitive way to understand a network's structure. We will use `matplotlib` to draw the graph.
    ```python
    plt.figure(figsize=(15, 10))
    # We can use the geographic coordinates stored in the nodes for a more meaningful layout
    pos = {node: (data['Longitude'], data['Latitude']) for node, data in G.nodes(data=True)}

    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, node_color='skyblue')
    plt.title("GEANT Central European Network Topology")
    plt.show()
    ```

**Step 4: Analyze Network Centrality (Find Critical Nodes)**
1.  **Centrality metrics** help us identify the most important or influential nodes in a graph. We will calculate **Betweenness Centrality**.
2.  **Betweenness Centrality** measures how many times a node lies on the shortest path between other pairs of nodes. A node with high betweenness centrality acts as a "bridge" and could be a major bottleneck or single point of failure.
    ```python
    # Calculate betweenness centrality for all nodes
    betweenness = nx.betweenness_centrality(G)

    # Sort the nodes by their centrality score
    sorted_centrality = sorted(betweenness.items(), key=lambda item: item[1], reverse=True)

    print("--- Top 5 Most Critical Nodes (by Betweenness Centrality) ---")
    for node, centrality in sorted_centrality[:5]:
        print(f"Node: {node}, Centrality Score: {centrality:.4f}")
    ```

**Step 5: Find the Most Efficient Path**
1.  A fundamental graph problem is finding the shortest path between two nodes. This is the basis of network routing.
2.  Let's find the shortest path between two major cities in the network, for example, Prague and Vienna.
    ```python
    # First, find the node IDs for Prague and Vienna from the node data
    prague_node = [node for node, data in G.nodes(data=True) if data.get('label') == 'Prague'][0]
    vienna_node = [node for node, data in G.nodes(data=True) if data.get('label') == 'Vienna'][0]

    # Calculate the shortest path using Dijkstra's algorithm (default in networkx)
    shortest_path = nx.shortest_path(G, source=prague_node, target=vienna_node)

    print(f"\n--- Shortest Path from Prague to Vienna ---")
    # Convert node IDs back to labels for readability
    path_labels = [G.nodes[node].get('label', node) for node in shortest_path]
    print(" -> ".join(path_labels))
    ```

**Step 6: Community Detection (Network Segmentation)**
1.  Community detection algorithms find tightly connected clusters of nodes that are less connected to the rest of the network. This is useful for understanding the logical segmentation of a network.
    ```python
    from networkx.algorithms import community

    # Use the Louvain community detection algorithm
    communities = community.louvain_communities(G)

    # Create a color map for visualization
    node_colors = {}
    for i, com in enumerate(communities):
        for node in com:
            node_colors[node] = i

    plt.figure(figsize=(15, 10))
    nx.draw(G, pos, with_labels=True, node_color=[node_colors.get(node, 0) for node in G.nodes()],
            node_size=50, font_size=8, cmap=plt.cm.jet)
    plt.title("Network Communities Detected by Louvain Algorithm")
    plt.show()
    ```
    *The visualization will now show distinct colored clusters, representing the detected network segments.*

#### **6. Success Criteria**
*   The team can successfully load a network from a `.graphml` file into a `networkx` graph object.
*   The team can generate and interpret a visualization of the network topology.
*   The team can calculate **betweenness centrality** and identify the top 5 most critical nodes in the network.
*   The team can find and display the **shortest path** between any two given nodes in the graph.
*   The team can apply a **community detection** algorithm and produce a visualization where nodes are colored according to their community assignment.

#### **7. Next Steps & Extensions**
*   **Weighted Paths:** The edges in many `graphml` files have weights (e.g., representing latency or distance). Modify the `nx.shortest_path` call to use these weights (`weight='latency'`) to find the "cheapest" path instead of just the one with the fewest hops.
*   **Graph-based Features for ML:** For each node, calculate a set of graph metrics (degree, centrality, clustering coefficient). This creates a feature set that can be used to train a machine learning model to predict node properties, such as its likelihood of failure.
*   **Simulate Network Failure:** Programmatically remove one of the critical nodes identified in Step 4 and analyze the impact. For example, check if the graph becomes disconnected or if the average shortest path length between all nodes increases significantly.

---
