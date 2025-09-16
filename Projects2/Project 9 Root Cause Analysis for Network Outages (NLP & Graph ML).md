---

### **Project 9: Root Cause Analysis for Network Outages (NLP & Graph ML)**

**Objective:** To build an automated system that analyzes a flood of network alerts, correlates them with the network topology, and identifies the most probable root cause device.

**Dataset:** **Synthetically Generated**. We will create a realistic network topology and simulate an alert storm caused by a single device failure. This allows us to have a "ground truth" to validate our algorithm's conclusion.

**Methodology:**
1.  **Topology Modeling:** Represent the network as a graph using the `networkx` library.
2.  **Alert Simulation:** Generate a list of realistic alert messages, including the root cause and its symptoms.
3.  **NLP Parsing:** Use regular expressions to extract the hostnames from alert messages.
4.  **Graph Analysis:** Calculate the **betweenness centrality** of the alerting nodes. The node that lies on the most shortest paths between other nodes is a strong candidate for the root cause, as its failure would have the widest impact.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 9: Root Cause Analysis for Network Outages (NLP & Graph ML)
# ==================================================================================
#
# Objective:
# This notebook demonstrates an automated approach to Root Cause Analysis by
# combining NLP for alert parsing and graph theory for topology analysis.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import random

# ----------------------------------------
# 2. Create the Network Topology and Simulate an Alert Storm
# ----------------------------------------
print("--- Step 1: Building Network Topology and Simulating Alert Storm ---")

# Define a realistic network topology (e.g., Core, Distribution, Access layers)
topology = {
    'Core-Router-1': ['Dist-Switch-A', 'Dist-Switch-B', 'Firewall'],
    'Firewall': ['Internet-Gateway'],
    'Dist-Switch-A': ['Access-Switch-1', 'Access-Switch-2'],
    'Dist-Switch-B': ['Access-Switch-3', 'Access-Switch-4'],
    'Access-Switch-1': ['Server-101', 'Server-102'],
    'Access-Switch-2': ['PC-201', 'PC-202'],
    'Access-Switch-3': ['Server-301', 'Server-302'],
    'Access-Switch-4': ['PC-401', 'PC-402']
}

# Create a graph from the topology
G = nx.Graph()
for node, neighbors in topology.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Function to simulate an alert storm
def generate_alert_storm(graph, root_cause_node):
    alerts = []
    # The root cause alert
    alerts.append(f"CRITICAL: Device {root_cause_node} interface Gi0/1 is down.")
    
    # Find all downstream nodes that are now unreachable from the 'Firewall' (our monitoring point)
    # To do this, we temporarily remove the faulty node from the graph
    temp_graph = graph.copy()
    temp_graph.remove_node(root_cause_node)
    
    for node in graph.nodes():
        # If a node is not the root cause itself and is no longer reachable from the monitor...
        if node != root_cause_node and not nx.has_path(temp_graph, 'Firewall', node):
            alert_type = random.choice(['is unreachable', 'failed to respond to ping', 'has high packet loss'])
            alerts.append(f"ERROR: Monitored device {node} {alert_type}.")
            
    return alerts

# --- SIMULATION ---
# Let's simulate a failure on a critical distribution switch
ROOT_CAUSE_DEVICE = 'Dist-Switch-A'
alert_messages = generate_alert_storm(G, ROOT_CAUSE_DEVICE)

print(f"\nSimulated a failure on: {ROOT_CAUSE_DEVICE}")
print("Generated Alert Storm (sample):")
for alert in random.sample(alert_messages, min(5, len(alert_messages))):
    print(f"- {alert}")


# ----------------------------------------
# 3. Parse Alerts and Identify Alerting Devices
# ----------------------------------------
print("\n--- Step 2: Parsing Alerts with NLP (Regex) ---")

def parse_hostnames(alerts):
    hostnames = set()
    # Regex to find device names that look like 'Word-Word-Number' or similar
    pattern = re.compile(r'([A-Za-z]+-[A-Za-z]+-\d+|[A-Za-z]+-\d+|[A-Za-z]+-[A-Za-z]+-[A-Za-z]+|\bFirewall\b|\bInternet-Gateway\b)')
    for alert in alerts:
        match = pattern.search(alert)
        if match:
            hostnames.add(match.group(0))
    return list(hostnames)

alerting_devices = parse_hostnames(alert_messages)
print(f"Identified {len(alerting_devices)} unique alerting devices:")
print(alerting_devices)


# ----------------------------------------
# 4. Root Cause Analysis using Graph Centrality
# ----------------------------------------
print("\n--- Step 3: Performing Root Cause Analysis via Graph Centrality ---")

# Calculate Betweenness Centrality for all nodes in the graph.
# This metric measures how many times a node acts as a bridge along the shortest
# path between two other nodes. A high value is indicative of a critical chokepoint.
centrality = nx.betweenness_centrality(G)

# Find the alerting device with the highest centrality score.
max_centrality = -1
predicted_root_cause = None

print("\nCentrality scores for alerting devices:")
for device in alerting_devices:
    score = centrality.get(device, 0)
    print(f"- {device}: {score:.4f}")
    if score > max_centrality:
        max_centrality = score
        predicted_root_cause = device

print(f"\nGround Truth Root Cause: {ROOT_CAUSE_DEVICE}")
print(f"Predicted Root Cause:    {predicted_root_cause}")

if predicted_root_cause == ROOT_CAUSE_DEVICE:
    print("\nSUCCESS: The algorithm correctly identified the root cause!")
else:
    print("\nFAILURE: The algorithm pointed to the wrong device.")


# ----------------------------------------
# 5. Visualization of the Result
# ----------------------------------------
print("\n--- Step 4: Visualizing the Result ---")

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42)

# Define colors for nodes
node_colors = []
for node in G.nodes():
    if node == predicted_root_cause:
        node_colors.append('red')      # Root Cause
    elif node in alerting_devices:
        node_colors.append('orange')   # Symptom
    else:
        node_colors.append('skyblue')  # Normal

nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2500, font_size=8, font_weight='bold')
plt.title('Network Topology with Identified Root Cause', size=15)
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Predicted Root Cause', markerfacecolor='red', markersize=15),
    plt.Line2D([0], [0], marker='o', color='w', label='Symptom Device', markerfacecolor='orange', markersize=15),
    plt.Line2D([0], [0], marker='o', color='w', label='Normal Device', markerfacecolor='skyblue', markersize=15)
]
plt.legend(handles=legend_handles, loc='best')
plt.show()


# ----------------------------------------
# 6. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("This notebook successfully demonstrated a powerful method for automated Root Cause Analysis.")
print("Key Takeaways:")
print(f"- We transformed {len(alert_messages)} potentially confusing alerts into a single, actionable insight: '{predicted_root_cause}' is the likely problem.")
print("- The combination of NLP (to extract entities from text) and Graph Theory (to understand system relationships) is a highly effective pattern for RCA.")
print("- The 'betweenness centrality' metric served as an excellent heuristic to find the most critical 'chokepoint' among the set of failing devices.")
print("- This approach can drastically reduce Mean Time To Resolution (MTTR) for network outages by directing engineers straight to the source of the issue.")
```