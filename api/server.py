import os
import uuid
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.settings import ConfigManager
from core.conversation import LongConversation, GeneralConversation
from prompts.templates import ConversationPrompts
from utils.logging import get_logger
from utils.aux import *

logger = get_logger("APIServer")

class ConversationManager:
    """Manages conversation sessions"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_dir = self.config.system.conversation_directory
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure conversation directory exists"""
        log_messages = []
        if not os.path.exists(self.conversation_dir):
            os.makedirs(self.conversation_dir)
            logger.info(f"Created conversation directory: {self.conversation_dir}")
            log_messages.append(f"Created conversation directory: {self.conversation_dir}")
        return log_messages
    
    def create_conversation(self, request: 'InitConversationRequest') -> Tuple[str, str, str]:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        log_messages = []
        logger.info(f"Creating new conversation with ID: {conversation_id}")
        log_messages.append(f"Creating new conversation with ID: {conversation_id}")
        
        # Get prompts
        prompts = ConversationPrompts()
        target = prompts.get("HYPERTENSION_CONSULTATION_TARGET")
        routine = prompts.get("HYPERTENSION_ASSESSMENT_ROUTINE")
        
        # Create patient info
        patient_info = {
            "name": request.name,
            "age": request.age,
            "gender": request.gender
        }
        
        # Add patient info to prompt
        patient_str = f"Patient information: Patient Name {request.name}, Age {request.age}, Gender {request.gender}"
        prompt = f"{target}\n{patient_str}"
        
        # Initialize conversation based on model type
        if request.model == "DrHyper":
            logger.info(f"Initializing conversation with DrHyper model for {request.name}")
            log_messages.append(f"Initializing conversation with DrHyper model for {request.name}")
            
            conv_model, graph_model = load_models(verbose=False)
            
            conv = LongConversation(
                target=prompt,
                conv_model=conv_model,
                graph_model=graph_model,
                routine=routine,
                visualize=False,
                weight_threshold=0.1,
                working_directory=self.config.system.working_directory,
            )
            
            # Load pre-built graphs if available
            entity_graph_path = os.path.join(self.config.system.working_directory, "entity_graph.pkl")
            relation_graph_path = os.path.join(self.config.system.working_directory, "relation_graph.pkl")

            if os.path.exists(entity_graph_path) and os.path.exists(relation_graph_path):
                logger.info(f"Loading existing graphs")
                log_messages.append(f"Loading existing graphs")
                graph_log_messages = conv.load_graph(entity_graph_path, relation_graph_path)
                log_messages.extend(graph_log_messages)
            else:
                logger.info(f"Initializing new graph")
                log_messages.append(f"Initializing new graph")
                graph_log_messages = conv.init_graph(save=True)
                log_messages.extend(graph_log_messages)
                
        else:
            error_msg = f"Unsupported model: {request.model}"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise ValueError(error_msg)
        
        # Initialize conversation
        logger.info("Initializing conversation...")
        log_messages.append("Initializing conversation...")
        ai_message, init_log_messages = conv.init()
        log_messages.extend(init_log_messages)
        
        # Store conversation
        self.conversations[conversation_id] = {
            "conv": conv,
            "messages": [{"role": "assistant", "content": ai_message}],
            "patient": patient_info,
            "model": request.model
        }
        
        logger.info(f"Created conversation {conversation_id} with model {request.model}")
        log_messages.append(f"Created conversation {conversation_id} with model {request.model}")
        
        return conversation_id, ai_message, "\n".join(log_messages)
    
    def process_message(self, conversation_id: str, human_message: str) -> Tuple[str, bool, str]:
        """Process a chat message"""
        log_messages = []
        
        if conversation_id not in self.conversations:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise ValueError(error_msg)
            
        conv_data = self.conversations[conversation_id]
        conv = conv_data["conv"]
        
        # Process message
        logger.info(f"Processing message in conversation {conversation_id}")
        log_messages.append(f"Processing message in conversation {conversation_id}")
        
        ai_response, accomplish, process_log_messages = conv.conversation(human_message)
        log_messages.extend(process_log_messages)
        
        # Update message history
        conv_data["messages"].append({"role": "user", "content": human_message})
        conv_data["messages"].append({"role": "assistant", "content": ai_response})
        
        logger.info(f"Processed message in conversation {conversation_id}, accomplish status: {accomplish}")
        log_messages.append(f"Processed message in conversation {conversation_id}, accomplish status: {accomplish}")
        
        return ai_response, accomplish, "\n".join(log_messages)
    
    def save_conversation(self, conversation_id: str, in_memory: bool = False):
        """Save conversation to disk"""
        log_messages = []
        
        if conversation_id not in self.conversations:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise ValueError(error_msg)
            
        conv_data = self.conversations[conversation_id]
        filepath = os.path.join(self.conversation_dir, f"{conversation_id}.pkl")
        
        try:
            with open(filepath, "wb") as f:
                pickle.dump(conv_data, f)
            
            # Remove from memory
            if in_memory is False:
                del self.conversations[conversation_id]
                
            logger.info(f"Saved conversation {conversation_id} to disk")
            log_messages.append(f"Saved conversation {conversation_id} to disk")
            
        except Exception as e:
            error_msg = f"Failed to save conversation {conversation_id}: {str(e)}"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise
            
        return "\n".join(log_messages)
    
    def load_conversation(self, conversation_id: str):
        """Load conversation from disk"""
        log_messages = []
        filepath = os.path.join(self.conversation_dir, f"{conversation_id}.pkl")
        
        if not os.path.exists(filepath):
            error_msg = f"Conversation file not found: {filepath}"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with open(filepath, "rb") as f:
                conv_data = pickle.load(f)
            
            self.conversations[conversation_id] = conv_data
            logger.info(f"Loaded conversation {conversation_id} from disk")
            log_messages.append(f"Loaded conversation {conversation_id} from disk")
            
        except Exception as e:
            error_msg = f"Failed to load conversation {conversation_id}: {str(e)}"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise
            
        return "\n".join(log_messages)
    
    def list_conversations(self):
        """List all conversations"""
        log_messages = []
        
        in_memory = list(self.conversations.keys())
        
        on_disk = []
        for filename in os.listdir(self.conversation_dir):
            if filename.endswith(".pkl"):
                conv_id = filename[:-4]
                on_disk.append(conv_id)
        
        logger.info(f"Listed {len(in_memory)} in-memory conversations and {len(on_disk)} on-disk conversations")
        log_messages.append(f"Listed {len(in_memory)} in-memory conversations and {len(on_disk)} on-disk conversations")
                
        return {"in_memory": in_memory, "on_disk": on_disk}, "\n".join(log_messages)
    
    def end_conversation(self, conversation_id: str, in_memory: bool = False):
        """End conversation"""
        log_messages = []
        
        if conversation_id not in self.conversations:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise ValueError(error_msg)
            
        conv_data = self.conversations[conversation_id]
        save_log_messages = self.save_conversation(conversation_id, in_memory=in_memory)
        log_messages.append(save_log_messages)
        
        logger.info(f"Ended conversation {conversation_id}")
        log_messages.append(f"Ended conversation {conversation_id}")
        
        return "\n".join(log_messages)
    
    def update_settings(self, component: str, parameter: str, value: Any):
        """
        Update system settings

        Args:
            component: Component to update (matching section names in config file like "SYSTEM", "GRAPH", "CONVERSATION LLM", "GRAPH LLM")
            parameter: Parameter name to update
            value: New value for the parameter

        Raises:
            ValueError: If component or parameter is invalid
        """
        log_messages = []
        
        # Normalize the component name to match config attributes
        component_attr = component.lower().replace(" ", "_")

        # Check if the component exists in the config
        if not hasattr(self.config, component_attr):
            error_msg = f"Component {component} not found in configuration. Valid components are: SYSTEM, GRAPH, CONVERSATION LLM, GRAPH LLM"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise ValueError(error_msg)

        # Get the component object
        config_component = getattr(self.config, component_attr)

        # Check if the parameter exists in the component
        if not hasattr(config_component, parameter):
            error_msg = f"Unknown parameter '{parameter}' for component {component}"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise ValueError(error_msg)

        # Update the parameter
        try:
            old_value = getattr(config_component, parameter)
            setattr(config_component, parameter, value)
            logger.info(f"Updated {component}.{parameter} from {old_value} to {value}")
            log_messages.append(f"Updated {component}.{parameter} from {old_value} to {value}")
        except Exception as e:
            error_msg = f"Failed to update {component}.{parameter} to {value}: {str(e)}"
            logger.error(error_msg)
            log_messages.append(error_msg)
            raise
            
        return "\n".join(log_messages)

# Initialize app and manager
app = FastAPI(title="DrHyper Conversation API")
manager = ConversationManager()

# Request/Response Models
class InitConversationRequest(BaseModel):
    name: str
    age: int
    gender: str
    model: str

class InitConversationResponse(BaseModel):
    conversation_id: str
    ai_message: str
    log_messages: str = ""

class ChatRequest(BaseModel):
    conversation_id: str
    human_message: str

class ChatResponse(BaseModel):
    ai_message: str
    accomplish: bool = False  # if the diagnosis is finished, accomplish the conversation
    log_messages: str = ""

class EndConversationRequest(BaseModel):
    conversation_id: str
    in_memory: bool = False  # whether to keep the conversation in memory after ending

class EndConversationResponse(BaseModel):
    conversation_id: str
    log_messages: str = ""

class SettingsUpdateRequest(BaseModel):
    component: str
    parameter: str
    value: Any

class SettingsUpdateResponse(BaseModel):
    message: str
    log_messages: str = ""

class ConversationResponse(BaseModel):
    message: str
    log_messages: str = ""

class ListConversationsResponse(BaseModel):
    in_memory: List[str]
    on_disk: List[str]
    log_messages: str = ""

# Endpoints
@app.post("/init_conversation", response_model=InitConversationResponse)
async def init_conversation(request: InitConversationRequest):
    """Initialize a new conversation"""
    try:
        conversation_id, ai_message, log_messages = manager.create_conversation(request)
        return InitConversationResponse(
            conversation_id=conversation_id,
            ai_message=ai_message, 
            log_messages=log_messages
        )
    except Exception as e:
        logger.error(f"Failed to initialize conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message"""
    try:
        ai_message, accomplish, log_messages = manager.process_message(request.conversation_id, request.human_message)
        return ChatResponse(
            ai_message=ai_message,
            accomplish=accomplish,
            log_messages=log_messages
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# this is for manually ending the conversation
@app.post("/end_conversation", response_model=EndConversationResponse)
async def end_conversation(request: EndConversationRequest):
    """End conversation and generate assessment"""
    try:    
        log_messages = manager.end_conversation(request.conversation_id, request.in_memory)
        
        return EndConversationResponse(
            conversation_id=request.conversation_id,
            log_messages=log_messages
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to end conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# manual save/load/list endpoints
@app.post("/save_conversation", response_model=ConversationResponse)
async def save_conversation(conversation_id: str):
    """Save conversation to disk"""
    try:
        log_messages = manager.save_conversation(conversation_id, in_memory=True)  # save conversation request will maintain the conversation in memory
        return ConversationResponse(
            message="Conversation saved successfully",
            log_messages=log_messages
        )
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_conversation", response_model=ConversationResponse)
async def load_conversation(conversation_id: str):
    """Load conversation from disk"""
    try:
        log_messages = manager.load_conversation(conversation_id)
        return ConversationResponse(
            message="Conversation loaded successfully",
            log_messages=log_messages
        )
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_conversations", response_model=ListConversationsResponse)
async def list_conversations():
    """List all conversations"""
    try:
        conversation_lists, log_messages = manager.list_conversations()
        return ListConversationsResponse(
            in_memory=conversation_lists["in_memory"],
            on_disk=conversation_lists["on_disk"],
            log_messages=log_messages
        )
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_settings", response_model=SettingsUpdateResponse)
async def update_settings(request: SettingsUpdateRequest):
    """Update system settings"""
    try:
        log_messages = manager.update_settings(request.component, request.parameter, request.value)
        return SettingsUpdateResponse(
            message="Settings updated successfully",
            log_messages=log_messages
        )
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))