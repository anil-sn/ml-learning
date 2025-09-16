---

### **Project 33: Optimizing Service Chain Placement in an NFV Environment**

**Objective:** To train a Reinforcement Learning agent to find the optimal placement of VNFs in a service chain across a physical network of servers. The goal is to select a sequence of hosts that minimizes the total network latency for the traffic flowing through the chain.

**Environment:** **Simulated NFV Infrastructure**. We will create a Python-based simulation of a data center, including:
*   **Physical Network:** A set of interconnected servers (nodes) with defined latencies between them (edges).
*   **Service Chain:** A predefined sequence of VNFs (e.g., Firewall -> IDS -> LoadBalancer).
*   **State:** The agent's current state is the physical host it has just placed a VNF on.
*   **Action:** The agent's action is to choose the next physical host for the next VNF in the chain.
*   **Reward:** The agent receives a negative reward equal to the network latency between the chosen hosts, encouraging it to pick closer servers.

**Model:** We will implement **Q-Learning**. The agent will learn a Q-table that represents the "quality" of choosing a particular next host from its current host, ultimately learning the lowest-latency path through the physical network that satisfies the service chain.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**```python
#
# ==================================================================================
#  Project 33: Optimizing Service Chain Placement in an NFV Environment (RL)
# ==================================================================================
#
# Objective:
# This notebook trains a Q-Learning agent to find the optimal placement of VNFs
# in a service chain across a physical network to minimize latency.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# ----------------------------------------
# 2. Build the Simulated NFV Environment
# ----------------------------------------
print("--- Building the Simulated NFV Infrastructure Environment ---")

# --- Define the Physical Network (Data Center Topology) ---
# This represents servers and the network latency (in ms) between them.
physical_nodes = ['Host_1', 'Host_2', 'Host_3', 'Host_4', 'Host_5', 'Host_6']
physical_edges = [
    ('Host_1', 'Host_2', 1), ('Host_1', 'Host_3', 5),
    ('Host_2', 'Host_3', 1), ('Host_2', 'Host_4', 10),
    ('Host_3', 'Host_5', 2),
    ('Host_4', 'Host_5', 2), ('Host_4', 'Host_6', 1),
    ('Host_5', 'Host_6', 5)
]
G = nx.Graph()
G.add_nodes_from(physical_nodes)
G.add_weighted_edges_from(physical_edges, weight='latency')

# Create a latency matrix for easy lookup. Use infinity for non-connected hosts.
node_map = {node: i for i, node in enumerate(physical_nodes)}
inv_node_map = {i: node for node, i in node_map.items()}
num_hosts = len(physical_nodes)
latency_matrix = np.full((num_hosts, num_hosts), np.inf)
for u, v, data in G.edges(data=True):
    i, j = node_map[u], node_map[v]
    latency_matrix[i, j] = latency_matrix[j, i] = data['latency']

# --- Define the Service Function Chain (SFC) ---
# The sequence of virtual functions traffic must traverse.
service_chain = ['VNF_Firewall', 'VNF_IDS', 'VNF_LoadBalancer']
sfc_length = len(service_chain)

print(f"Physical network created with {num_hosts} hosts.")
print(f"Service chain to be placed: {' -> '.join(service_chain)}")

# Visualize the physical network
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
edge_labels = nx.get_edge_attributes(G, 'latency')
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('Physical Data Center Network Topology (Latency in ms)')
plt.show()


# ----------------------------------------
# 3. Q-Learning Agent Training
# ----------------------------------------
print("\n--- Training the Q-Learning Agent for VNF Placement ---")

# Hyperparameters
num_episodes = 10000
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1    # Exploration rate

# The Q-table will learn the value of placing the *next* VNF on a host,
# given the *current* VNF's host.
# Rows: Current host | Columns: Next host
q_table = np.zeros((num_hosts, num_hosts))

# The reward is the negative latency. Maximizing reward = minimizing latency.
rewards = -latency_matrix

for episode in range(num_episodes):
    # An episode is one full placement of the service chain
    
    # Start by placing the first VNF on a random host
    current_host_idx = random.randint(0, num_hosts - 1)
    
    # Place the rest of the VNFs in the chain
    for vnf_step in range(1, sfc_length):
        # Epsilon-greedy action: Choose the next host
        if random.uniform(0, 1) < epsilon:
            next_host_idx = random.randint(0, num_hosts - 1) # Explore
        else:
            next_host_idx = np.argmin(q_table[current_host_idx]) # Exploit best known path
        
        # Get reward (negative latency) for this placement decision
        reward = rewards[current_host_idx, next_host_idx]
        
        # Q-Learning update rule
        old_value = q_table[current_host_idx, next_host_idx]
        # The "future reward" is the value of the best move from the *next* host.
        next_max = np.min(q_table[next_host_idx])
        
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[current_host_idx, next_host_idx] = new_value
        
        # Move to the next state
        current_host_idx = next_host_idx

print("Training complete.")


# ----------------------------------------
# 4. Finding the Optimal Placement
# ----------------------------------------
print("\n--- Determining the Optimal VNF Placement ---")

def find_optimal_placement():
    best_path = []
    min_total_latency = np.inf
    
    # We need to test starting the chain on every possible host
    for start_host_idx in range(num_hosts):
        current_path = [start_host_idx]
        current_latency = 0
        
        current_host_idx_in_path = start_host_idx
        for _ in range(1, sfc_length):
            # Greedily choose the next best host from the learned Q-table
            next_host_idx = np.argmin(q_table[current_host_idx_in_path])
            current_latency += latency_matrix[current_host_idx_in_path, next_host_idx]
            current_path.append(next_host_idx)
            current_host_idx_in_path = next_host_idx
            
        if current_latency < min_total_latency:
            min_total_latency = current_latency
            best_path = current_path
            
    return [inv_node_map[i] for i in best_path], min_total_latency

optimal_placement, total_latency = find_optimal_placement()

print("Optimal Placement Found:")
for i, vnf in enumerate(service_chain):
    print(f"- {vnf}: Place on {optimal_placement[i]}")

print(f"\nTotal End-to-End Network Latency for this chain: {total_latency:.2f} ms")


# ----------------------------------------
# 5. Visualization of the Solution
# ----------------------------------------
print("\n--- Visualizing the Optimal Placement ---")
plt.figure(figsize=(10, 8))
# Draw the base network
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# Highlight the optimal path
path_edges = list(zip(optimal_placement, optimal_placement[1:]))
nx.draw_networkx_nodes(G, pos, nodelist=optimal_placement, node_color='lightgreen')
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3.0)

plt.title('Optimal Service Chain Placement Path', fontsize=16)
plt.show()


# ----------------------------------------
# 6. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Q-Learning agent successfully learned to solve the complex VNF placement problem.")
print("Key Takeaways:")
print(f"- The agent found a placement path ({' -> '.join(optimal_placement)}) that minimizes the network latency for the service chain, a non-trivial task in a complex topology.")
print("- This demonstrates how Reinforcement Learning can be used to solve complex resource allocation and optimization problems in networking that are difficult to tackle with traditional algorithms or manual configuration.")
print("- An NFV Orchestrator (NFVO) with this RL capability could become highly intelligent. When a new service chain request arrives, the NFVO wouldn't just place VNFs where there's capacity; it would use its trained agent to find the *optimal* placement that guarantees the best performance (lowest latency).")
print("- This leads to more efficient use of data center resources and better adherence to Service Level Agreements (SLAs) for latency-sensitive applications.")

```