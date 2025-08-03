# DrHyper: An Autonomous AI Physician for Hypertension Diagnosis

## Table of Contents
- [Project Introduction](#project-introduction)
- [Key Features](#key-features)
- [Technical Overview](#technical-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [API Server](#api-server)
- [Examples](#examples)
- [License](#license)

## Project Introduction

DrHyper is the first autonomous AI physician specifically designed for hypertension diagnosis, addressing the critical global shortage of medical professionals. With approximately 1.3 billion adults worldwide affected by hypertension, DrHyper provides an accessible, expert-level diagnostic solution that achieves 91.50% accuracy on Chinese hypertension specialty practice questions (CHSPQ) and 88.90% on international hypertension specialty practice questions (IHSPQ), surpassing both human physicians and general-purpose AI models.

Unlike traditional AI diagnostic systems that merely assist physicians with preexisting data, DrHyper autonomously conducts multi-turn patient conversations, systematically gathering personalized clinical information through an average of 26.41 dialogue turns while maintaining over 95% clinical entity acquisition rate.

## Key Features

- **Autonomous Patient Interaction**: Conducts comprehensive medical consultations without human intervention
- **Superior Diagnostic Accuracy**: Outperforms human specialists on standardized hypertension examinations
- **Extended Conversational Capability**: Maintains coherent dialogues averaging 26+ turns (vs. 4-6 for comparison models)
- **Graph-Based Information Management**: Uses dual-graph architecture for tracking clinical entities and relationships
- **Entropy-Based Knowledge Retrieval**: Implements MinRAG methodology for efficient medical knowledge access
- **Multi-Language Support**: Supports both English and Chinese for global accessibility

## Technical Overview

### Core Technologies

1. **Multi-Stage Training Pipeline**
   - Reinforcement Learning with Grouped Preference Optimization (GRPO)
   - Supervised Fine-Tuning with Chain-of-Thought (CoT) reasoning
   - Entropy-based Minimal RAG (MinRAG) for knowledge injection

2. **Zero-Shot Task-Oriented Conversation Framework**
   - Dynamic dual-graph architecture (entity graph + relation graph)
   - Community detection using Leiden algorithm
   - Strategic node selection based on uncertainty quantification

3. **Information Persistence Mechanisms**
   - Conversation Memory Capacity (CMC): 6.25 (vs. 4.10-4.36 for comparison models)
   - Information Retrieval Capacity (IRC): 1.66 (vs. 0.96-1.07 for comparison models)
   - 97.5% information concordance rate throughout extended dialogues

### Model Architecture

DrHyper is built on DeepSeek-R1-Distill-Qwen models (7B and 32B variants) enhanced with:
- Medical domain-specific knowledge from 254,385 clinical records
- 2,641 physician-annotated chain-of-thought instances
- 33 authoritative hypertension textbooks and guidelines

## Installation

### Prerequisites

- Python 3.12 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- At least 16GB RAM (32GB recommended)

### Clone the Repository

```bash
git clone https://github.com/yourusername/DrHyper.git
cd DrHyper
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Pre-trained Models (Optional)

DrHyper framework requires two large language models:
1. Graph-LLM, responsible for:
   - Generating entity and relation graphs focused on the conversation target (hypertension diagnosis by default)
   - Updating graph topology and nodes based on extracted patient response information

2. Conversation-LLM, responsible for:
   - Providing AI responses to patient queries or generating questions based on graph information
   - Extracting information from patient (user) responses
   - Generating hint messages for conversation guidance according to graph information

You can configure these models following the instructions in the [Configuration](#configuration) section below. The framework supports both local models and online API integrations. Our pre-trained DrHyper models are available for download from our HuggingFace repository:
```
https://huggingface.co/jeff010913/DrHyper_Family
```

## Configuration

DrHyper uses a configuration file located at `./config/config.cfg`. Here's a detailed explanation of each section:

### Conversation LLM Configuration

```ini
[CONVERSATION LLM]
provider = custom          # Options: openai, custom, local
api_key = your-api-key    # Your API key for the LLM provider
base_url = https://api-endpoint.com  # API endpoint URL
model = model-name        # Model identifier
model_path = ""           # Path for local models (if provider=local)
max_tokens = 8192         # Maximum tokens for response
temperature = 0.8         # Response creativity (0.0-1.0)
```

### Graph LLM Configuration

```ini
[GRAPH LLM]
provider = custom          # Same options as conversation LLM
api_key = your-api-key    
base_url = https://api-endpoint.com
model = model-name
model_path = ""
max_tokens = 8192
temperature = 0.8
```

### System Configuration

```ini
[SYSTEM]
working_directory = ./artifacts           # Directory for model artifacts
conversation_directory = ./conversations  # Directory for conversation logs
language = 中文                            # 
stream = False                            # Enable streaming responses
```

### Graph Parameters

```ini
[GRAPH]
node_hit_threshold = 3      # Max queries per node to avoid repetition
confidential_threshold = 0.2 # Threshold for entity confidentiality (0-1)
relevance_threshold = 0.2    # Threshold for entity relevance (0-1)
weight_threshold = 0.8       # Threshold for entity importance (0-1)

# Node selection parameters
alpha = 1.0   # Weight for node importance
beta = 1.0    # Weight for topology importance  
gamma = 1.0   # Penalty for cluster transfer
```

## Usage

### Command Line Interface

DrHyper provides a CLI for both graph creation and patient consultations.

#### Create Knowledge Graph

To create and save the knowledge graph without starting a conversation:

```bash
python cli.py create-graph --verbose
```

This will:
- Initialize the entity and relation graphs
- Save graphs to the configured working directory
- Display creation progress and file locations

#### Start a Conversation

To start an interactive patient consultation:

```bash
python cli.py start --graph-dir ./artifacts --verbose
```

Options:
- `--graph-dir`: Directory containing pre-built graphs (optional)
- `--verbose`: Enable detailed debug output
- `--no-color`: Disable colored terminal output

During the conversation:
- Type your responses as a patient would
- Type `exit`, `quit`, or `bye` to end the conversation
- Press Ctrl+C to interrupt

### API Server

DrHyper provides a FastAPI-based server for integration with other applications.

#### Start the Server

```bash
python server.py
```

Or with custom host/port:

```bash
python deploy.py --host 0.0.0.0 --port 8000 --reload
```

#### API Endpoints

##### 1. Initialize Conversation

```http
POST /init_conversation
Content-Type: application/json

{
    "name": "John Doe",
    "age": 55,
    "gender": "male",
    "model": "Dr.Hyper"  // Options: "Dr.Hyper", "LLAMA", "DeepSeek", "Qwen"
}
```

Response:
```json
{
    "conversation_id": "uuid-string",
    "ai_message": "Initial greeting and first question"
}
```

##### 2. Send Message

```http
POST /chat
Content-Type: application/json

{
    "conversation_id": "uuid-string",
    "human_message": "Patient's response"
}
```

Response:
```json
{
    "ai_message": "AI doctor's response"
}
```

##### 3. End Conversation

```http
POST /end_conversation
Content-Type: application/json

{
    "conversation_id": "uuid-string",
    "patient_background_condition": "Optional background info",
    "patient_background_diagnosis": "Optional diagnosis info"
}
```

Response:
```json
{
    "conversation_id": "uuid-string",
    "assessment": "Clinical assessment",
    "patient_condition": "Summarized condition",
    "patient_diagnosis": "Final diagnosis"
}
```

## Examples

### Example 1: Command Line Consultation

```bash
# First, create the knowledge graph
$ python cli.py create-graph --verbose
System: Loading AI models...
System: Initializing knowledge graph...
System: Graph creation completed successfully!
System: Entity graph saved to: ./artifacts/entity_graph.pkl
System: Relation graph saved to: ./artifacts/relation_graph.pkl

# Start a consultation
$ python cli.py start --verbose
System: Found existing graphs:
System: Entity graph: ./artifacts/entity_graph.pkl
System: Relation graph: ./artifacts/relation_graph.pkl
System: No patient info found. You need to provide patient information:
Patient name: Zhang Wei
Patient age: 58
Patient gender (male/female/other): male
System: Initializing conversation...

Dr.Hyper: Hello Mr. Zhang Wei, I'm Dr. Hyper, your AI hypertension specialist. I understand you're here for a hypertension consultation. To provide you with the best care, I'll need to ask you some questions about your health. Let's start with your blood pressure. Have you measured your blood pressure recently? If yes, what were the readings?

Patient: Yes, I measured it yesterday. It was 145/92.

Dr.Hyper: Thank you for sharing that information. A blood pressure reading of 145/92 mmHg is indeed elevated. The systolic pressure (145) and diastolic pressure (92) both exceed normal ranges. How long have you been noticing elevated blood pressure readings? Have you been monitoring it regularly?

Patient: I've been checking it for about 2 months now. It's usually around 140-150 over 90-95.
```

### Example 2: API Integration

```python
import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Initialize conversation
init_data = {
    "name": "Li Ming",
    "age": 62,
    "gender": "male",
    "model": "Dr.Hyper"
}

response = requests.post(f"{BASE_URL}/init_conversation", json=init_data)
result = response.json()
conversation_id = result["conversation_id"]
print(f"AI: {result['ai_message']}")

# Send patient messages
messages = [
    "My blood pressure has been high lately, around 150/95",
    "I've been feeling dizzy in the mornings",
    "Yes, I take medication but sometimes forget",
    "I don't exercise much and eat a lot of salty food"
]

for message in messages:
    chat_data = {
        "conversation_id": conversation_id,
        "human_message": message
    }
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    print(f"Patient: {message}")
    print(f"AI: {response.json()['ai_message']}\n")

# End conversation and get assessment
end_data = {
    "conversation_id": conversation_id
}
response = requests.post(f"{BASE_URL}/end_conversation", json=end_data)
assessment = response.json()
print(f"Final Assessment: {assessment['assessment']}")
print(f"Diagnosis: {assessment['patient_diagnosis']}")
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
