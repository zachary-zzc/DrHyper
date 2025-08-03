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
from utils.llm_loader import load_chat_model
from utils.logging import get_logger, setup_conversation_logger

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
        if not os.path.exists(self.conversation_dir):
            os.makedirs(self.conversation_dir)
    
    def create_conversation(self, request: 'InitConversationRequest') -> Tuple[str, str]:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        conv_logger, mem_handler = setup_conversation_logger(conversation_id)
        
        # Get prompts
        prompts = ConversationPrompts()
        target = prompts.get("HYPERTENSION_TARGET")
        routine = prompts.get("ROUTINE")
        
        # Create patient info
        patient_info = {
            "name": request.name,
            "age": request.age,
            "gender": request.gender
        }
        
        # Build prompt
        patient_str = prompts.get("PATIENT_INFO", 
            name=request.name, age=request.age, gender=request.gender)
        prompt = f"{target}\n{patient_str}"
        
        # Initialize conversation based on model type
        if request.model == "Dr.Hyper":
            chat_model = load_chat_model("ali_api|qwen-max-latest")
            graph_model = load_chat_model("ali_api|qwen-max-latest")
            
            # Add reference standards
            prompt += "\n\n" + prompts.get("REFERENCE_STANDARDS")
            
            conv = LongConversation(
                target=prompt,
                chat_model=chat_model,
                graph_model=graph_model,
                routine=routine,
                visualize=False,
                weight_threshold=0.1
            )
            
            # Load pre-built graphs if available
            entity_graph_path = os.path.join("artifacts", "entity_graph.pkl")
            relation_graph_path = os.path.join("artifacts", "relation_graph.pkl")
            
            if os.path.exists(entity_graph_path) and os.path.exists(relation_graph_path):
                conv.load_graph(entity_graph_path, relation_graph_path)
            else:
                conv.init_graph(save=True)
                
        else:
            # Load appropriate model
            model_map = {
                "LLAMA": "ali_api|llama3.1-405b-instruct",
                "DeepSeek": "ali_api|deepseek-r1",
                "Qwen": "ali_api|qwen-max-latest"
            }
            
            if request.model not in model_map:
                raise ValueError(f"Unsupported model: {request.model}")
                
            chat_model = load_chat_model(model_map[request.model])
            
            # Add routine guidance
            prompt += f"\n\n{prompts.get('GENERAL_GUIDANCE', routine=routine)}"
            
            conv = GeneralConversation(
                prompt=prompt,
                chat_model=chat_model
            )
        
        # Initialize conversation
        ai_message = conv.init()
        
        # Store conversation
        self.conversations[conversation_id] = {
            "conv": conv,
            "log": mem_handler.logs,
            "messages": [{"role": "assistant", "content": ai_message}],
            "patient": patient_info,
            "model": request.model
        }
        
        logger.info(f"Created conversation {conversation_id} with model {request.model}")
        return conversation_id, ai_message
    
    def process_message(self, conversation_id: str, human_message: str) -> str:
        """Process a chat message"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
            
        conv_data = self.conversations[conversation_id]
        conv = conv_data["conv"]
        
        # Process message
        ai_response = conv.conversation(human_message)
        
        # Update message history
        conv_data["messages"].append({"role": "user", "content": human_message})
        conv_data["messages"].append({"role": "assistant", "content": ai_response})
        
        logger.info(f"Processed message in conversation {conversation_id}")
        return ai_response
    
    def save_conversation(self, conversation_id: str):
        """Save conversation to disk"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
            
        conv_data = self.conversations[conversation_id]
        filepath = os.path.join(self.conversation_dir, f"{conversation_id}.pkl")
        
        with open(filepath, "wb") as f:
            pickle.dump(conv_data, f)
            
        # Remove from memory
        del self.conversations[conversation_id]
        logger.info(f"Saved conversation {conversation_id} to disk")
    
    def load_conversation(self, conversation_id: str):
        """Load conversation from disk"""
        filepath = os.path.join(self.conversation_dir, f"{conversation_id}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Conversation file not found: {filepath}")
            
        with open(filepath, "rb") as f:
            conv_data = pickle.load(f)
            
        # Re-attach logger
        conv_logger, mem_handler = setup_conversation_logger(conversation_id)
        mem_handler.logs = conv_data.get("log", [])
        
        self.conversations[conversation_id] = conv_data
        logger.info(f"Loaded conversation {conversation_id} from disk")
    
    def list_conversations(self) -> Dict[str, List[str]]:
        """List all conversations"""
        in_memory = list(self.conversations.keys())
        
        on_disk = []
        for filename in os.listdir(self.conversation_dir):
            if filename.endswith(".pkl"):
                conv_id = filename[:-4]
                on_disk.append(conv_id)
                
        return {"in_memory": in_memory, "on_disk": on_disk}
    
    def end_conversation(self, conversation_id: str, background: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """End conversation and generate assessment"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
            
        conv_data = self.conversations[conversation_id]
        return conv_data       

# Initialize app and manager
app = FastAPI(title="Medical Conversation API")
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

class ChatRequest(BaseModel):
    conversation_id: str
    human_message: str

class ChatResponse(BaseModel):
    ai_message: str

class EndConversationRequest(BaseModel):
    conversation_id: str
    patient_background_condition: Optional[str] = None
    patient_background_diagnosis: Optional[str] = None

class EndConversationResponse(BaseModel):
    conversation_id: str
    assessment: str
    patient_condition: str
    patient_diagnosis: str

# Endpoints
@app.post("/init_conversation", response_model=InitConversationResponse)
async def init_conversation(request: InitConversationRequest):
    """Initialize a new conversation"""
    try:
        conversation_id, ai_message = manager.create_conversation(request)
        return InitConversationResponse(
            conversation_id=conversation_id,
            ai_message=ai_message
        )
    except Exception as e:
        logger.error(f"Failed to initialize conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message"""
    try:
        ai_message = manager.process_message(request.conversation_id, request.human_message)
        return ChatResponse(ai_message=ai_message)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end_conversation", response_model=EndConversationResponse)
async def end_conversation(request: EndConversationRequest):
    """End conversation and generate assessment"""
    try:
        background = {}
        if request.patient_background_condition:
            background["condition"] = request.patient_background_condition
        if request.patient_background_diagnosis:
            background["diagnosis"] = request.patient_background_diagnosis
            
        result = manager.end_conversation(request.conversation_id, background)
        
        return EndConversationResponse(
            conversation_id=request.conversation_id,
            assessment=result["assessment"],
            patient_condition=result["patient_condition"],
            patient_diagnosis=result["patient_diagnosis"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to end conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_assessment")
async def download_assessment(conversation_id: str):
    """Download assessment report"""
    try:
        if conversation_id not in manager.conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        assessment = manager.conversations[conversation_id].get("assessment")
        if not assessment:
            raise HTTPException(status_code=404, detail="Assessment not found")
            
        # Save assessment to file
        assessment_dir = "assessments"
        os.makedirs(assessment_dir, exist_ok=True)
        
        filepath = os.path.join(assessment_dir, f"{conversation_id}_assessment.json")
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(assessment, f, ensure_ascii=False, indent=4)
            
        return FileResponse(
            path=filepath,
            media_type="application/json",
            filename=f"{conversation_id}_assessment.json"
        )
    except Exception as e:
        logger.error(f"Failed to download assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_conversation")
async def save_conversation(conversation_id: str):
    """Save conversation to disk"""
    try:
        manager.save_conversation(conversation_id)
        return {"message": "Conversation saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_conversation")
async def load_conversation(conversation_id: str):
    """Load conversation from disk"""
    try:
        manager.load_conversation(conversation_id)
        return {"message": "Conversation loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_conversations")
async def list_conversations():
    """List all conversations"""
    return manager.list_conversations()

@app.post("/update_language")
async def update_language(language: str):
    """Update system language"""
    if language not in ["en", "zh"]:
        raise HTTPException(status_code=400, detail="Language must be 'en' or 'zh'")
        
    config = ConfigManager()
    config.update_language(language)
    return {"message": f"Language updated to {language}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)