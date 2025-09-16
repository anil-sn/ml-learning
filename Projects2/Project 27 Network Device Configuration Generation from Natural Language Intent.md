---

### **Project 27: Network Device Configuration Generation from Natural Language Intent**

**Objective:** To build an NLP model that can parse a natural language command from a network engineer (e.g., "Block traffic from the guest vlan to the database server") and extract the key entities (Source, Destination, Action) to create a structured firewall rule.

**Dataset Source:** **Synthetically Generated**. We will create our own dataset of natural language commands and the corresponding structured rules. This is a common and necessary practice in specialized domains where public datasets are not available.

**Model:** We will use a pre-trained **Named Entity Recognition (NER)** model from the popular `spaCy` library. NER is a perfect fit for this task, as it's designed to find and label specific entities (like 'PERSON' or 'ORGANIZATION') in text. We will teach it to find custom network entities like 'SOURCE_IP', 'DEST_PORT', and 'ACTION'.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**```python
#
# ==================================================================================
#  Project 27: Network Device Configuration Generation from Natural Language Intent
# ==================================================================================
#
# Objective:
# This notebook builds a proof-of-concept intent-based networking system by using
# a spaCy NER model to parse natural language and generate structured firewall rules.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Install spaCy and Download a Model
# ----------------------------------------
print("--- Installing spaCy and a pre-trained model ---")
!pip install -q spacy
!python -m spacy download en_core_web_sm

import spacy
from spacy.tokens import Span
from spacy.training import Example

# ----------------------------------------
# 2. Synthetic Training Data Generation
# ----------------------------------------
print("\n--- Generating Synthetic Training Data for Custom NER ---")

# We need to provide examples of text and the entities we want to find.
# The format is (text, {'entities': [(start_char, end_char, LABEL)]})
TRAIN_DATA = [
    ("Block traffic from 192.168.1.10 to 10.0.0.5 on port 443",
     {'entities': [(0, 5, 'ACTION'), (19, 30, 'SOURCE_IP'), (34, 43, 'DEST_IP'), (54, 57, 'DEST_PORT')]}),
    ("Allow access from the guest vlan to the internet on port 80",
     {'entities': [(0, 5, 'ACTION'), (20, 31, 'SOURCE_ZONE'), (35, 47, 'DEST_ZONE'), (58, 60, 'DEST_PORT')]}),
    ("Deny all connections from host 172.16.30.5",
     {'entities': [(0, 4, 'ACTION'), (29, 41, 'SOURCE_IP')]}),
    ("Permit tcp traffic from any to the web server on port 443",
     {'entities': [(0, 6, 'ACTION'), (24, 27, 'SOURCE_ZONE'), (31, 43, 'DEST_ZONE'), (54, 57, 'DEST_PORT')]}),
    ("block traffic from the marketing network to the finance server",
     {'entities': [(0, 5, 'ACTION'), (20, 37, 'SOURCE_ZONE'), (41, 56, 'DEST_ZONE')]}),
    ("let host 10.1.1.1 access the dmz",
     {'entities': [(0, 3, 'ACTION'), (9, 18, 'SOURCE_IP'), (26, 29, 'DEST_ZONE')]}),
    ("drop connections from vlan 100 to server 192.168.100.200",
     {'entities': [(0, 4, 'ACTION'), (22, 30, 'SOURCE_ZONE'), (34, 58, 'DEST_IP')]})
]
print(f"Created {len(TRAIN_DATA)} training examples.")


# ----------------------------------------
# 3. Training a Custom NER Model
# ----------------------------------------
print("\n--- Training a Custom NER Model ---")

# Load a pre-trained model to start from (transfer learning)
nlp = spacy.load("en_core_web_sm")
# Get the NER component
ner = nlp.get_pipe("ner")

# Add our new custom labels to the NER model
labels = ['ACTION', 'SOURCE_IP', 'DEST_IP', 'SOURCE_ZONE', 'DEST_ZONE', 'DEST_PORT']
for label in labels:
    ner.add_label(label)

# The training loop
optimizer = nlp.create_optimizer()
for i in range(20): # Loop 20 times over the data
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
    if i % 5 == 0:
        print(f"Iteration {i}, Losses: {losses}")
print("Training complete.")


# ----------------------------------------
# 4. Building the Intent Parser
# ----------------------------------------
print("\n--- Building the Intent Parser Function ---")

def parse_intent(text):
    """
    Takes a natural language command and returns a structured dictionary.
    """
    doc = nlp(text.lower()) # Process the text with our custom model
    
    # Initialize a dictionary to hold the rule components
    rule = {
        'action': 'deny', # Default action
        'source': 'any',
        'destination': 'any',
        'port': 'any'
    }
    
    print(f"\nProcessing command: '{text}'")
    print("Detected Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")
        if ent.label_ in ['ACTION']:
            action_text = ent.text.lower()
            if action_text in ['allow', 'permit', 'let']:
                rule['action'] = 'permit'
            else:
                rule['action'] = 'deny'
        elif ent.label_ in ['SOURCE_IP', 'SOURCE_ZONE']:
            rule['source'] = ent.text
        elif ent.label_ in ['DEST_IP', 'DEST_ZONE']:
            rule['destination'] = ent.text
        elif ent.label_ in ['DEST_PORT']:
            rule['port'] = ent.text
            
    return rule

def generate_config_from_rule(rule):
    """
    Takes a structured rule and generates a pseudo-firewall config line.
    """
    return f"access-list 101 {rule['action']} ip {rule['source']} {rule['destination']} eq {rule['port']}"


# ----------------------------------------
# 5. Testing the System
# ----------------------------------------
print("\n--- Testing the Intent-Based System ---")

# Define some test commands that the model has NOT seen before
test_commands = [
    "I need to block the host 192.168.50.1 from accessing the internet",
    "please permit traffic from the guest vlan to the printer on port 631",
    "deny access from 10.20.30.40 to our web server",
    "let anyone from the user network get to the dmz"
]

for command in test_commands:
    # Parse the natural language to get a structured rule
    parsed_rule = parse_intent(command)
    print("Parsed Rule:", parsed_rule)
    
    # Generate the configuration from the structured rule
    generated_config = generate_config_from_rule(parsed_rule)
    print("Generated Config:", generated_config)
    print("-" * 30)


# ----------------------------------------
# 6. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("This notebook successfully demonstrated a proof-of-concept for intent-based networking.")
print("Key Takeaways:")
print("- By training a custom Named Entity Recognition (NER) model, we were able to teach the system to understand the specific language of network engineering.")
print("- The system can reliably extract key entities (like IPs, zones, and actions) from unstructured text and convert them into a structured, machine-readable format.")
print("- This structured data can then be used to automatically generate device configurations, reducing the chance of human error (like typos in an IP address) and ensuring consistency.")
print("- While this is a simplified example, it represents the future of network management. In a full-scale system, the generated rule would be sent to an automation controller (like Ansible or NSO), which would then apply the configuration to the actual network devices, completing the loop from human intent to network reality.")
```