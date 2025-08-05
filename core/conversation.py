import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from langchain.schema import AIMessage, SystemMessage, HumanMessage, BaseMessage
from html.parser import HTMLParser

from config.settings import ConfigManager
from .graph import EntityGraph
from utils.logging import get_logger

class ThinkParser(HTMLParser):
    """Parser for extracting think tags from AI responses"""
    def __init__(self):
        super().__init__()
        self.think_content = []
        self.clean_content = []
        self.in_think_tag = False
        self.all_content = []
        self.found_closing_think = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "think":
            self.in_think_tag = True

    def handle_endtag(self, tag):
        if tag.lower() == "think":
            self.found_closing_think = True
            self.in_think_tag = False

    def handle_data(self, data):
        self.all_content.append(data)
        if self.in_think_tag:
            self.think_content.append(data)
        else:
            self.clean_content.append(data)

class BaseConversation:
    """Base conversation class with common functionality"""
    
    def __init__(self, chat_model, max_tokens: int = 8192):
        self.config = ConfigManager()
        self.chat_model = chat_model
        self.messages: List[BaseMessage] = []
        self.think_history: List[Dict[str, Any]] = []
        self.max_tokens = max_tokens
        # self.logger = get_logger(self.__class__.__name__)
        
    def _process_response(self, response_text: str) -> Dict[str, str]:
        """Extract think content and regular response from AI response"""
        parser = ThinkParser()
        parser.feed(response_text)
        
        # Handle case where there's a closing </think> tag but no opening tag
        if parser.found_closing_think and not parser.think_content:
            end_think_index = response_text.lower().find("</think>")
            if end_think_index > 0:
                parser.think_content = [response_text[:end_think_index]]
                parser.clean_content = [response_text[end_think_index + 8:]]
        
        return {
            "response": "".join(parser.clean_content).strip(),
            "think": "".join(parser.think_content).strip()
        }

class LongConversation(BaseConversation):
    """Long conversation with graph-based entity tracking"""
    
    def __init__(
        self,
        target: str,
        conv_model,
        graph_model,
        routine: Optional[str] = None,
        visualize: bool = False,
        working_directory: Optional[str] = None,
        stream: bool = False,
        **graph_params
    ):
        super().__init__(conv_model)
        self.target = target
        self.graph_model = graph_model
        self.routine = routine
        self.visualize = visualize
        self.working_directory = working_directory or self.config.system.working_directory
        self.stream = stream
        
        # Initialize entity graph with configuration parameters
        self.plan_graph = EntityGraph(
            target=target,
            graph_model=graph_model,
            conv_model=conv_model,
            routine=routine,
            visualize=visualize,
            working_directory=self.working_directory,
            node_hit_threshold=graph_params.get('node_hit_threshold', self.config.system.node_hit_threshold),
            confidential_threshold=graph_params.get('confidential_threshold', self.config.system.confidential_threshold),
            relevance_threshold=graph_params.get('relevance_threshold', self.config.system.relevance_threshold),
            weight_threshold=graph_params.get('weight_threshold', self.config.system.weight_threshold),
            alpha=graph_params.get('alpha', self.config.system.alpha),
            beta=graph_params.get('beta', self.config.system.beta),
            gamma=graph_params.get('gamma', self.config.system.gamma)
        )
        
        self.current_hint = ""
        self.messages: List[BaseMessage] = []
        self.entire_messages: List[BaseMessage] = []
        self.think_history: List[Dict[str, Any]] = []
        self.message_reserve_turns: int = 2  # Number of turns to reserve in history
        
        self._ensure_working_directory()
    
    def _ensure_working_directory(self):
        """Ensure working directory exists"""
        log_messages = []
        if self.working_directory and not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
            # self.logger.info(f"Created working directory: {self.working_directory}")
            log_messages.append(f"Created working directory: {self.working_directory}")
        else:
            log_messages.append("Working directory already exists or not specified")
        return log_messages
    
    def init_graph(self, save: bool = False):
        """Initialize the entity graph"""
        # self.logger.info("Initializing entity graph...")
        log_messages = self.plan_graph.init(save=save)
        # self.logger.info("\n".join(log_messages))
        return log_messages
    
    def load_graph(self, entity_graph_path: str, relation_graph_path: str):
        """Load existing graphs from files"""
        # self.logger.info(f"Loading graphs from {entity_graph_path} and {relation_graph_path}")
        log_messages = self.plan_graph.load_graphs(entity_graph_path, relation_graph_path)
        # self.logger.info("\n".join(log_messages))
        return log_messages
    
    def init(self):
        """Initialize the conversation and return the first AI message"""
        log_messages = []
        # self.logger.info("Initializing conversation...")
        log_messages.append("Initializing conversation...")
        
        hint_message, plan_status, hint_log_messages = self.plan_graph.get_hint_message()
        log_messages.extend(hint_log_messages)
        
        self.messages.append(SystemMessage(content=hint_message))
        
        # self.logger.info(f"Initial hint message: {hint_message}")
        log_messages.append(f"Initial hint message: {hint_message[:100]}...")
        
        response = self.chat_model.invoke(self.messages, stream=self.stream)
        
        processed_response = self._process_response(response.content)
        response_content = processed_response["response"]
        think_content = processed_response["think"]
        
        if think_content:
            self.think_history.append({
                "turn": 0,
                "think": think_content,
            })
            # self.logger.debug(f"Think content: {think_content}")
            log_messages.append(f"Captured think content of length: {len(think_content)}")

        
        self.messages.pop()  # Remove the hint message
        self.messages.append(AIMessage(content=response_content))
        # Store the entire message history for later reference, the entire conversations do not contain hint messages
        self.entire_messages.append(AIMessage(content=response_content))
        self.current_hint = hint_message
        
        # self.logger.info(f"Conversation initialized, initial response length: {len(response_content)}")
        log_messages.append(f"Conversation initialized, initial response length: {len(response_content)}")
        
        return response_content, log_messages
    
    def _get_message_turns(self) -> int:
        """Get the number of turns in the conversation"""
        return len(self.messages) // 2
    
    def conversation(self, human_message: str):
        """Process a conversation turn and return AI response"""
        log_messages = []
        # self.logger.info("Processing conversation turn...")
        log_messages.append("Processing conversation turn...")
        
        # Get the last AI message for context
        query_message = self.messages[-1].content if self.messages else ""
        
        # Update graph with new information
        # self.logger.info("Updating graph with new information...")
        update_log_messages = self.plan_graph.accept_message(self.current_hint, query_message, human_message)
        log_messages.extend(update_log_messages)
        
        # Get new hint
        # self.logger.info("Getting new hint message...")
        hint_message, plan_status, hint_log_messages = self.plan_graph.get_hint_message()
        log_messages.extend(hint_log_messages)
        
        self.current_hint = hint_message
        
        # Prepare messages for this turn
        # reserve the last messages according to the message_reserve_turns
        if self._get_message_turns() > self.message_reserve_turns:
            original_len = len(self.messages)
            self.messages = self.messages[-(self.message_reserve_turns * 2):]
            # self.logger.info(f"Trimmed message history from {original_len} to {len(self.messages)}")
            log_messages.append(f"Trimmed message history from {original_len} to {len(self.messages)}")
            
        self.messages.append(HumanMessage(content=human_message))
        self.messages.append(SystemMessage(content=hint_message))
        # Store the human message to the entire conversation messages for later reference
        self.entire_messages.append(HumanMessage(content=human_message))
        
        # self.logger.info(f"Conversation turn with hint: {hint_message[:100]}...")
        log_messages.append(f"Conversation turn with hint: {hint_message[:100]}...")
        
        response = self.chat_model.invoke(self.messages, stream=self.stream)
        
        # Process response
        turn_index = len(self.think_history) + 1
        processed_response = self._process_response(response.content)
        response_content = processed_response["response"]
        think_content = processed_response["think"]
        
        if think_content:
            self.think_history.append({
                "turn": turn_index,
                "think": think_content,
            })
            # self.logger.debug(f"Think content: {think_content}")
            log_messages.append(f"Captured think content of length: {len(think_content)}")
        
        self.messages.pop()  # Remove hint message
        self.messages.append(AIMessage(content=response_content))
        self.entire_messages.append(AIMessage(content=response_content))

        # self.logger.info(f"Conversation turn completed, response length: {len(response_content)}")
        log_messages.append(f"Conversation turn completed, response length: {len(response_content)}")
        
        if self.plan_graph.accomplish:
            # self.logger.info("Conversation goal accomplished!")
            log_messages.append("Conversation goal accomplished!")

        # return response content, accomplishment status, and log messages
        return response_content, self.plan_graph.accomplish, log_messages

class GeneralConversation(BaseConversation):
    """General conversation without graph tracking"""
    
    def __init__(self, prompt: str, chat_model, working_directory: Optional[str] = None, stream: bool = False):
        """
        Initialize the general conversation class.
        """
        super().__init__(chat_model)
        self.prompt = prompt
        self.working_directory = working_directory
        self.stream = stream
        
        log_messages = self._ensure_working_directory()
        # self.logger.info("\n".join(log_messages))

    def _ensure_working_directory(self):
        """Ensure working directory exists"""
        log_messages = []
        if self.working_directory and not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
            # self.logger.info(f"Created working directory: {self.working_directory}")
            log_messages.append(f"Created working directory: {self.working_directory}")
        else:
            log_messages.append("Working directory already exists or not specified")
        return log_messages

    def init(self):
        """Initialize conversation with system prompt"""
        log_messages = []
        # self.logger.info("Initializing general conversation...")
        log_messages.append("Initializing general conversation...")
        
        self.messages.append(SystemMessage(content=self.prompt))
        response = self.chat_model.invoke(self.messages, stream=self.stream)
        self.messages.append(response)
        
        # self.logger.info(f"Conversation initialized, initial response length: {len(response.content)}")
        log_messages.append(f"Conversation initialized, initial response length: {len(response.content)}")
        
        return response.content, log_messages
    
    def conversation(self, human_message: str):
        """Process a conversation turn"""
        log_messages = []
        # self.logger.info("Processing conversation turn...")
        log_messages.append("Processing conversation turn...")
        
        self.messages.append(HumanMessage(content=human_message))
        response = self.chat_model.invoke(self.messages, stream=self.stream)
        self.messages.append(response)
        
        # self.logger.info(f"Conversation turn completed, response length: {len(response.content)}")
        log_messages.append(f"Conversation turn completed, response length: {len(response.content)}")
        
        return response.content, log_messages