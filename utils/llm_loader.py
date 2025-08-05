from typing import List, Optional, Dict, Any, Iterator, Union
import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from threading import Thread

from config.settings import ConfigManager

class CustomChatModel(BaseChatModel):
    """Custom Chat Model that interfaces with OpenAI API compatible custom LLMs"""
    
    model_name: str = None
    api_key: str = None
    base_url: str = None
    temperature: float = 0.0
    max_tokens: int = 8192
    
    def __init__(self, model_name, api_key, base_url, temperature, max_tokens, **kwargs):
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "custom_api"

    def _convert_message(self, msg: BaseMessage):
        if isinstance(msg, HumanMessage):
            return {"role": "user", "content": msg.content}
        elif isinstance(msg, AIMessage):
            return {"role": "assistant", "content": msg.content}
        elif isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")

    def _create_client(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        client = self._create_client()
        
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[self._convert_message(msg) for msg in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            stop=stop,
        )
        
        return completion.choices[0].message.content

    def _stream_response(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Iterator[str]:
        """Stream the LLM response one token at a time."""
        client = self._create_client()
        
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=[self._convert_message(msg) for msg in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            stop=stop,
        )
        
        for chunk in stream:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    if run_manager:
                        run_manager.on_llm_new_token(content)
                    yield content
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatResult:
        if not stream:
            content = self._call(messages, stop=stop, **kwargs)
            generation = ChatGeneration(message=AIMessage(content=content))
            
            if run_manager:
                run_manager.on_llm_new_token(generation.message.content)
                
            return ChatResult(generations=[generation])
        else:
            # Instead of returning a generator, collect the entire response
            content = ""
            for token in self._stream_response(messages, stop=stop, run_manager=run_manager, **kwargs):
                content += token
                
            # Return a complete ChatResult with the full content
            generation = ChatGeneration(message=AIMessage(content=content))
            return ChatResult(generations=[generation])


class LocalChatModel(BaseChatModel):
    """Chat Model that loads and runs local model files (e.g. from Hugging Face)"""
    
    model_path: str = None
    model_type: str = "causal"  # causal or other model architectures
    temperature: float = 0.0
    max_tokens: int = 1024
    device: str = "auto"  # 'cpu', 'cuda', 'auto'
    model_kwargs: Dict[str, Any] = {}
    generation_kwargs: Dict[str, Any] = {}
    
    _model = None
    _tokenizer = None
    _pipeline = None
    
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        
        # Use configuration values as defaults if not explicitly provided
        self.model_path = model_path
        self.temperature = kwargs.get('temperature', ConfigManager().system.temperature)
        self.max_tokens = kwargs.get('max_tokens', ConfigManager().system.max_tokens)
        self.device = kwargs.get('device', 'auto')  # Default to 'auto' for device selection
        
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer from the specified path"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map=self.device,
            **self.model_kwargs
        )
        
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device_map=self.device
        )
    
    @property
    def _llm_type(self) -> str:
        return "local_model"

    def _format_chat_history(self, messages: List[BaseMessage]) -> str:
        """Format the chat history into a single string for the model"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_messages.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                formatted_messages.append(f"System: {msg.content}")
            else:
                formatted_messages.append(f"Other: {msg.content}")
        
        return "\n".join(formatted_messages) + "\nAssistant: "

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        if self._pipeline is None:
            self._load_model()
        
        prompt = self._format_chat_history(messages)
        
        generation_config = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
            **self.generation_kwargs
        }
        
        if stop:
            generation_config["stopping_criteria"] = stop
        
        outputs = self._pipeline(
            prompt,
            **generation_config
        )
        
        generated_text = outputs[0]["generated_text"]
        
        # Remove the prompt from the generated text
        response = generated_text[len(prompt):]
        
        # Check if any stop sequence appears in the response and truncate
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response[:response.index(stop_seq)]
        
        return response.strip()

    def _stream_response(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Iterator[str]:
        """Stream the LLM response one token at a time."""
        if self._model is None or self._tokenizer is None:
            self._load_model()
        
        prompt = self._format_chat_history(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        # Create a streamer for token-wise generation
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_config = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
            **self.generation_kwargs
        }
        
        # Run generation in a separate thread
        generation_kwargs = dict(
            inputs=inputs,
            streamer=streamer,
            **generation_config
        )
        
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        
        accumulated_text = ""
        stop_found = False
        
        # Yield tokens from the streamer
        for new_text in streamer:
            accumulated_text += new_text
            
            # Check for stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in accumulated_text:
                        # Yield text up to the stop sequence
                        remaining = accumulated_text[:accumulated_text.index(stop_seq)]
                        if remaining:
                            if run_manager:
                                run_manager.on_llm_new_token(remaining)
                            yield remaining
                        stop_found = True
                        break
                
                if stop_found:
                    break
            
            if run_manager:
                run_manager.on_llm_new_token(new_text)
            yield new_text

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatResult, Iterator[ChatResult]]:
        if not stream:
            content = self._call(messages, stop=stop, **kwargs)
            generation = ChatGeneration(message=AIMessage(content=content))
            
            if run_manager:
                run_manager.on_llm_new_token(generation.message.content)
                
            return ChatResult(generations=[generation])
        else:
            # For streaming mode, return an iterator that yields partial results
            def generate_stream():
                content_so_far = ""
                for token in self._stream_response(messages, stop=stop, run_manager=run_manager, **kwargs):
                    content_so_far += token
                    yield ChatResult(generations=[
                        ChatGeneration(message=AIMessage(content=content_so_far))
                    ])
            
            return generate_stream()


def load_chat_model(provider: str, 
                    model_name: str = "", 
                    api_key: str = "",
                    base_url: str = "",
                    model_path: str = "",
                    temperature: float = 0.0,
                    max_tokens: int = 8192,
                    device: str = "auto") -> BaseChatModel:
    """Load a chat model by its fully specified name
    """
    if provider == "custom": 
        return CustomChatModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == "local":
        return LocalChatModel(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            device=device
        )
    else:
        return init_chat_model(model_name, model_provider=provider)