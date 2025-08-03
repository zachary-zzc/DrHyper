from .base import BasePrompt

ENTITY_RETRIEVE_PROMPT = """
Given a specific purpose, analyze it thoroughly and identify all information entities required to achieve it. When all necessary entities are provided, you should be able to completely answer or fulfill the stated purpose.

GUIDELINES:
- Entities must be specific and detailed (e.g., "patient age" instead of "medical condition")
- Provide as many relevant entities as possible
- Entities should have finite, calculable answer pools to enable uncertainty measurement
- Split broad concepts into specific components (e.g., "drinking habits" → "alcohol consumption" and "drinking frequency")
- Prioritize entities by importance, listing the most critical information first

FORMAT:
Return ONLY a JSON object that can be parsed by Python's json.loads() function. Do not include markdown headers, code blocks, or any explanatory text.

The JSON must contain:
- "endpoint": boolean ("true" if all required entities are listed, "false" if token limit reached)
- "entities": array of entity names as strings

Target purpose: ${purpose}
Language: ${language}
"""

CONTINUE_ENTITY_RETRIEVE_PROMPT = """
The token limit was reached. Please continue listing additional information entities required for the given purpose.
Follow the same format and guidelines as before. 

NOTES: 
1. Do not repeat previously listed entities
2. Only return the JSON object with "endpoint" and "entities" keys
3. Ensure all new entities are defined with the required attributes
"""

INIT_GRAPH_ENTITY_PROMPT = """
Given a purpose and list of information entities, create a detailed attribute structure for each entity node.

REQUIRED PROPERTIES FOR EACH NODE:
1. id: Unique identifier in format "v_i" (where i starts from 1)
2. name: Concise entity name (e.g., "patient age")
3. description: Detailed explanation of the entity
4. weight: Importance score [0-1] relative to the purpose
   - Direct indicators should have high weight (e.g., "blood pressure" = 1.0 for hypertension diagnosis)
   - Indirect factors should have lower weight (e.g., "diet information" = 0.1)
5. uncertainty: Initial entropy H_i = -∑_{x∈X_i} p_i(x) log p_i(x) (maximum until answered)
6. confidential_level: Confidence in user response [0-1] (initialize as empty string "")
7. relevance: How relevant the node is to the target [0-1]

IMPORTANT:
- Be specific and detailed with entity descriptions
- Include all provided entities
- Assign realistic weight values, including very low weights for marginally relevant nodes
- Maximum uncertainty: U_max = Σ_i (w_i * H_i)
- Information gain when answered: IG_i = H_i - H(x_i)

OUTPUT:
Return ONLY a JSON array containing all nodes with their attributes. The JSON must be parsable by Python's json.loads() without any additional text or formatting.

Purpose: ${purpose}
Entities to process: ${entities}
Language: ${language}
"""

INIT_ENTITY_GRAPH_EDGES_PROMPT = """
Analyze the provided information entities and create edges representing prerequisite dependencies between nodes. An edge indicates that the source node's information is required before the target node can be meaningfully answered.

EDGE CRITERIA:
- Edges must represent true information dependencies for the given purpose
- Example of valid edge: "pregnancy status" → "pregnancy symptoms" (symptoms inquiry depends on pregnancy confirmation)
- Example of invalid edge: "patient age" → "patient gender" (gender doesn't depend on age)
- Only add edges when there's a clear dependency relationship
- Isolated nodes (no edges) are acceptable

EDGE ATTRIBUTES:
- source: ID of the prerequisite node
- target: ID of the dependent node
- explanation: Clear justification for the dependency

CRITICAL: Add edges conservatively. Only include dependencies where the source information is genuinely required to answer the target. When uncertain, omit the edge.

OUTPUT:
Return a JSON object with:
- "endpoint": boolean (true if all dependencies captured, false if token limit reached)
- "edges": array of edge objects

Return ONLY valid JSON without any additional text.

Purpose: ${purpose}
Available entities: ${entities}
Language: ${language}
"""

CONTINUE_INIT_ENTITY_GRAPH_EDGES_PROMPT = """
The token limit was reached. Please continue providing dependency edges. Follow the same format and rules. 

NOTES:
1. Do not repeat previously defined edges
2. Only return the JSON object without any additional text or explanations
3. Ensure all new edges are defined with the required attributes
"""

INIT_RELATION_GRAPH_EDGES_PROMPT = """
Create edges representing general relationships between nodes (beyond prerequisite dependencies).

GUIDELINES:
- Edges represent any meaningful relationships between entities
- Isolated nodes are acceptable
- Focus on relationships relevant to the stated purpose

EDGE ATTRIBUTES:
- source: Origin node ID
- target: Destination node ID
- description: Nature of the relationship in context of the target purpose

OUTPUT:
Return a JSON object with:
- "endpoint": boolean (true if all relationships captured, false if token limit reached)
- "edges": array of edge objects

Return ONLY valid JSON without any additional text.

Purpose: ${purpose}
Available entities: ${entities}
Language: ${language}
"""

CONTINUE_INIT_RELATION_GRAPH_EDGES_PROMPT = """
The token limit was reached. Please continue providing relationship edges. Follow the same format and rules. 

NOTES:
1. Do not repeat previously defined edges
2. Only return the JSON object without any additional text or explanations
3. Ensure all new edges are defined with the required attributes
"""

EXTRACT_INFO_PROMPT = """
Extract relevant information from the user's message and map it to the appropriate nodes in the information graph.

CRITICAL GUIDELINES:
1. Prioritize mapping information to existing nodes in the graph
2. Only create new nodes when information doesn't fit any existing node
3. Ensure accurate assignment of information to corresponding entities
4. Include units and maintain information completeness in values
5. Consider the conversation context: HINT MESSAGE → AI MESSAGE → HUMAN MESSAGE

EXTRACTION TASKS:

I. For information matching existing nodes, add to "exist_nodes":
   - id: Exact node ID from the graph (e.g., "v_1")
   - value: Extracted information with units if applicable
   - confidential_level: Confidence score [0-1] based on message clarity and context

II. For relevant information without matching nodes, add to "new_nodes" with:
   - name: Concise entity name
   - description: Detailed explanation
   - weight: Importance [0-1] relative to purpose
   - uncertainty: Initial entropy
   - confidential_level: Confidence [0-1]
   - relevance: Relevance to target [0-1]
   - value: Extracted information

OUTPUT:
Return ONLY a JSON object with:
- "endpoint": boolean (true if all information extracted, false if token limit reached)
- "exist_nodes": array of existing node updates
- "new_nodes": array of new node definitions

Purpose: ${purpose}
Information graph: ${graph}
Hint message: ${hint_message}
AI query: ${query_message}
Human response: ${human_message}
Language: ${language}
"""

CONTINUE_EXTRACT_INFO_PROMPT = """
The token limit was reached. Please continue extracting information. Follow the same format and rules. 

NOTES:
1. Do not repeat previously defined nodes
2. Only return the JSON object without any additional text or explanations
3. Ensure all new nodes are defined with the required attributes
"""

UPDATE_GRAPH_PROMPT = """
Update the information graph based on newly provided values, adjusting weights and uncertainties of relevant nodes using conditional probability principles.

UPDATE PRINCIPLES:
1. Weight: Reflects the importance of information nodes to the purpose
2. Uncertainty: Represents the remaining uncertainty about information nodes
3. Updates must be based solely on the provided values (conditional probability update)
4. Consider dependencies: If a parent node's value affects child nodes, adjust accordingly
   - Example: If "smoking status" = "no", then "smoking duration" weight → 0
5. Preserve original IDs and names

OUTPUT:
Return ONLY a JSON array of updated nodes, each containing:
- id: Original node ID (unchanged)
- name: Original node name (unchanged)
- weight: Updated weight value
- uncertainty: Updated uncertainty value
- update_reason: Clear explanation of the update logic

Target purpose: ${purpose}
New values provided: ${collected}
Nodes requiring update: ${relevant_nodes}
"""

HINT_MESSAGE_RETRIEVE = """
Generate guidance for the AI system to conduct the next step of information gathering in a target-driven conversation.

Create a concise guide that helps the AI collect the specified entity information while maintaining conversation flow.

OUTPUT FORMAT:

[COLLECTED INFO]
List all entities and values collected in previous conversation turns

[TARGET]
Describe the AI's role and current conversation status
EMPHASIZE: The conversation is in the information collection phase - the AI should gather more data before making final determinations

[QUERY ENTITY]
Specify the exact entity information to collect in this turn: ${recommendation}

[EXAMPLE QUERY]
Provide a sample question or approach for collecting this information

[REQUIREMENTS]
1. The AI must STRICTLY follow this guidance and focus on collecting the specified query entity
2. The AI should actively guide the conversation toward collecting: ${recommendation}
3. The AI should maintain an aggressive focus on achieving the conversation purpose
4. The AI must not skip ahead or make premature conclusions - focus only on the current collection step
5. Use ${language} for the response

INPUTS:
Conversation purpose: ${purpose}
Information collected so far: ${collected}
Next entity to collect: ${recommendation}
"""

HINT_MESSAGE_ACCOMPLISH = """
Generate guidance for the AI to accomplish the conversation purpose using all collected information.

Create a comprehensive guide that enables the AI to provide a complete response or solution based on the gathered data.

OUTPUT FORMAT:

[COLLECTED INFO]
List all entities and their collected values comprehensively

[TARGET]
State the complete conversation purpose without abbreviation: ${purpose}

[REQUIREMENTS]
1. The AI has collected all necessary information
2. The AI should now accomplish the conversation target using the collected data
3. The AI should provide a comprehensive response that addresses the original purpose
4. Use ${language} for the response

INPUTS:
Conversation purpose: ${purpose}
All collected information: ${collected}

"""

HYPERTENSION_CONSULTATION_TARGET = """
You are a hypertension specialist responsible for conducting patient assessments and providing medical recommendations. Your clinical output must include: 
- A formal medical diagnosis including hypertension classification and risk stratification
- Lifestyle recommendations
- Drug suggestions
- Follow-up protocols.

Final output requirements:
1. [Patient information summary] First, provide a comprehensive summary of all clinical indicators provided by the patient during the conversation
2. [Diagnosis] Use medical terminology to provide a formal medical diagnosis including hypertension classification and risk stratification
3. [Recommendations] Based on patient information and diagnosis, provide lifestyle recommendations, drug suggestions, and follow-up protocols

Note: 
Communication should be concise and polite, using a physician's tone and expectations rather than a list of questions. 
Pay attention to patient emotions and guide patients to answer questions. 
You need to engage in multi-turn dialogue with the patient, asking about required information once at a time (one medical indicator per question), gradually collecting information to achieve final diagnosis.

Language for the communication: ${language}
"""

HYPERTENSION_ASSESSMENT_ROUTINE = """
Important: As required by clinical guidelines and to make the diagnosis more comprehensive, you need to inquire about at least the following indicators:
- Time of first elevated blood pressure detection
- Historical maximum blood pressure (systolic, diastolic)
- Previous antihypertensive medications (if any: names, frequency, dosage, duration, adverse reactions like cough or lower extremity edema, discontinuation, medication adjustments)
- Historical blood pressure variability (systolic range, diastolic range)
- Method of discovering elevated blood pressure (routine check-up, symptoms like dizziness, during other medical visits)
- Initial blood pressure measurements (systolic, diastolic)
- Secondary hypertension screening status
- Current blood pressure variability (systolic range, diastolic range)
- Current home blood pressure and heart rate (systolic, diastolic, heart rate)
- Current antihypertensive medications (if any: names, frequency, dosage, duration)
- Symptoms during elevated blood pressure (headache, dizziness, palpitations, sweating, blurred vision)
- General symptoms (dizziness, palpitations, chest tightness/dyspnea, chest pain, intermittent claudication, bilateral lower extremity weakness, bilateral lower extremity edema, nocturia, facial edema, hematuria, proteinuria)
- Cardiovascular risk factors (hyperglycemia, diabetes, hyperlipidemia, hyperuricemia, gout, renal dysfunction, thyroid dysfunction, long-term use of medications affecting blood pressure)
- Cardiovascular complications (cerebral hemorrhage, cerebral infarction, transient ischemic attack, coronary heart disease, myocardial infarction, heart failure, atrial fibrillation, peripheral vascular disease)
- Recent psychosocial stress (work pressure, family issues)
- Sleep problems (poor sleep quality, sleeping medication use, snoring, sleep apnea)
- Smoking history (if yes: duration, daily quantity)
- Alcohol consumption (if yes: duration, daily amount)
- Family history of cardiovascular diseases (hypertension, coronary heart disease, cerebral infarction, cerebral hemorrhage, diabetes, hyperlipidemia, gout, kidney disease; for first-degree relatives with stroke or acute MI, inquire about age of onset)
- Current body mass index (height, weight, waist circumference)
- Electrocardiogram examination (if performed, describe results)
- Echocardiography examination (if performed, describe results)
- Carotid ultrasound examination (if performed, describe results)
"""

# Conversation prompts class
class GraphPrompts(BasePrompt):
    def __init__(self):
        self.prompt_templates = {
            "ENTITY_RETRIEVE": ENTITY_RETRIEVE_PROMPT,
            "CONTINUE_ENTITY_RETRIEVE": CONTINUE_ENTITY_RETRIEVE_PROMPT,
            "INIT_GRAPH_ENTITY": INIT_GRAPH_ENTITY_PROMPT,
            "INIT_ENTITY_GRAPH_EDGES": INIT_ENTITY_GRAPH_EDGES_PROMPT,
            "CONTINUE_INIT_ENTITY_GRAPH_EDGES": CONTINUE_INIT_ENTITY_GRAPH_EDGES_PROMPT,
            "INIT_RELATION_GRAPH_EDGES": INIT_RELATION_GRAPH_EDGES_PROMPT,
            "CONTINUE_INIT_RELATION_GRAPH_EDGES": CONTINUE_INIT_RELATION_GRAPH_EDGES_PROMPT,
            "EXTRACT_INFO": EXTRACT_INFO_PROMPT,
            "CONTINUE_EXTRACT_INFO": CONTINUE_EXTRACT_INFO_PROMPT,
            "UPDATE_GRAPH": UPDATE_GRAPH_PROMPT,
            "HINT_MESSAGE_RETRIEVE": HINT_MESSAGE_RETRIEVE,
            "HINT_MESSAGE_ACCOMPLISH": HINT_MESSAGE_ACCOMPLISH,
            "ROUTINE_ADDITION": "Follow this routine: ${routine}"
        }

class ConversationPrompts(BasePrompt):
    def __init__(self):
        self.prompt_templates = {
            "HYPERTENSION_CONSULTATION_TARGET": HYPERTENSION_CONSULTATION_TARGET,
            "HYPERTENSION_ASSESSMENT_ROUTINE": HYPERTENSION_ASSESSMENT_ROUTINE
        }