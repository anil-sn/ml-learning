# Project 027: Network Device Configuration Generation from Natural Language Intent

## Objective
Build an NLP model that can parse natural language commands from network engineers (e.g., "Block traffic from the guest vlan to the database server") and extract key entities (Source, Destination, Action) to create structured firewall rules.

## Business Value
- **Intent-Based Networking**: Enable network engineers to configure devices using natural language instead of complex CLI commands
- **Reduced Human Error**: Minimize configuration mistakes by automatically translating intent to precise device commands
- **Faster Deployment**: Accelerate network changes by eliminating the need to manually write configuration syntax
- **Accessibility**: Allow non-expert staff to perform basic network configuration tasks
- **Consistency**: Ensure standardized configuration patterns across the network infrastructure

## Core Libraries
- **spaCy**: Pre-trained NLP models and Named Entity Recognition (NER) capabilities
- **scikit-learn**: Machine learning utilities for model evaluation and data processing
- **pandas**: Data manipulation and synthetic dataset generation
- **matplotlib/seaborn**: Visualization of NER results and model performance

## Dataset
- **Source**: Synthetically Generated
- **Size**: Custom training examples of natural language network commands
- **Features**: Network entities (ACTION, SOURCE_IP, DEST_IP, SOURCE_ZONE, DEST_ZONE, DEST_PORT)
- **Labels**: Entity boundaries and types for NER training
- **Type**: Text data with entity annotations for supervised NLP training

## Step-by-Step Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv network_nlp_env
source network_nlp_env/bin/activate  # On Windows: network_nlp_env\Scripts\activate

# Install required packages
pip install spacy pandas matplotlib seaborn
python -m spacy download en_core_web_sm
```

### 2. Create Training Data
```python
# Define synthetic training examples with entity annotations
TRAIN_DATA = [
    ("Block traffic from 192.168.1.10 to 10.0.0.5 on port 443",
     {'entities': [(0, 5, 'ACTION'), (19, 30, 'SOURCE_IP'), 
                   (34, 43, 'DEST_IP'), (54, 57, 'DEST_PORT')]}),
    ("Allow access from the guest vlan to the internet on port 80",
     {'entities': [(0, 5, 'ACTION'), (20, 31, 'SOURCE_ZONE'), 
                   (35, 47, 'DEST_ZONE'), (58, 60, 'DEST_PORT')]}),
    # Add more training examples...
]
```

### 3. Model Training
```python
import spacy
from spacy.training import Example

# Load pre-trained model and add custom NER labels
nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")

# Add custom network entity labels
labels = ['ACTION', 'SOURCE_IP', 'DEST_IP', 'SOURCE_ZONE', 'DEST_ZONE', 'DEST_PORT']
for label in labels:
    ner.add_label(label)

# Training loop
optimizer = nlp.create_optimizer()
for i in range(20):  # Multiple training iterations
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, sgd=optimizer)
```

### 4. Intent Parsing Function
```python
def parse_intent(text):
    """Parse natural language command and extract network entities"""
    doc = nlp(text.lower())
    
    rule = {
        'action': 'deny',  # Default action
        'source': 'any',
        'destination': 'any', 
        'port': 'any'
    }
    
    for ent in doc.ents:
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
```

### 5. Configuration Generation
```python
def generate_config_from_rule(rule):
    """Generate firewall configuration from structured rule"""
    return f"access-list 101 {rule['action']} ip {rule['source']} {rule['destination']} eq {rule['port']}"

# Example usage
command = "Block the host 192.168.50.1 from accessing the internet"
parsed_rule = parse_intent(command)
config = generate_config_from_rule(parsed_rule)
print(f"Generated: {config}")
```

### 6. Model Testing and Validation
```python
# Test commands the model hasn't seen
test_commands = [
    "permit traffic from the guest vlan to the printer on port 631",
    "deny access from 10.20.30.40 to our web server",
    "let anyone from the user network get to the dmz"
]

for command in test_commands:
    parsed_rule = parse_intent(command)
    generated_config = generate_config_from_rule(parsed_rule)
    print(f"Command: {command}")
    print(f"Config: {generated_config}")
```

## Success Criteria
- **High Entity Recognition Accuracy (>90%)**: Correctly identify network entities in natural language
- **Consistent Rule Generation**: Produce valid configuration commands from parsed intents
- **Robust Parsing**: Handle variations in natural language expressions
- **Extensible Design**: Easy to add new entity types and configuration templates

## Next Steps & Extensions
1. **Multi-vendor Support**: Extend to generate configs for different router/firewall vendors
2. **Complex Rules**: Support more sophisticated policies with multiple conditions
3. **Validation**: Add config syntax validation before deployment
4. **Integration**: Connect with network automation tools (Ansible, NAPALM)
5. **Web Interface**: Build user-friendly web interface for non-technical users
6. **Voice Commands**: Add speech-to-text for hands-free network configuration

## Files Structure
```
027_Network_Device_Config_Generation/
├── README.md
├── network_config_generation.ipynb
├── requirements.txt
└── models/
    └── (trained spaCy model files)
```

## Running the Project
1. Install spaCy and download the English model
2. Execute the Jupyter notebook step by step
3. Test with your own natural language commands
4. Extend training data for domain-specific terminology

This project demonstrates how NLP can revolutionize network management by making configuration accessible through natural language interfaces, reducing complexity and human error in network operations.