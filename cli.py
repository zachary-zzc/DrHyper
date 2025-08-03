# cli.py
import argparse
import os
import sys
import pickle

from core.conversation import LongConversation
from prompts.templates import ConversationPrompts
from utils.llm_loader import load_chat_model
from utils.logging import get_logger
from config.settings import ConfigManager

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"

def format_doctor_response(text: str) -> str:
    """Format doctor's response with color"""
    return f"{Colors.GREEN}{Colors.BOLD}Dr.Hyper:{Colors.RESET} {text}"

def format_patient_input(text: str) -> str:
    """Format patient input with color"""
    return f"{Colors.BLUE}{Colors.BOLD}Patient:{Colors.RESET} {text}"

def format_system_message(text: str) -> str:
    """Format system messages with color"""
    return f"{Colors.YELLOW}System:{Colors.RESET} {text}"

def format_debug(text: str) -> str:
    """Format debug messages with color"""
    return f"{Colors.CYAN}Debug:{Colors.RESET} {text}"

def format_error(text: str) -> str:
    """Format error messages with color"""
    return f"{Colors.RED}Error:{Colors.RESET} {text}"

def get_patient_info():
    """Get patient information from user input"""
    print(format_system_message("Please provide patient information:"))
    name = input("Patient name: ")
    
    while True:
        try:
            age = int(input("Patient age: "))
            if 0 < age < 120:
                break
            print(format_system_message("Please enter a valid age (1-120)"))
        except ValueError:
            print(format_system_message("Please enter a valid number"))
    
    gender = ""
    while gender.lower() not in ["male", "female", "other"]:
        gender = input("Patient gender (male/female/other): ")
        if gender.lower() not in ["male", "female", "other"]:
            print(format_system_message("Please enter 'male', 'female', or 'other'"))
            
    return name, age, gender

def load_models(verbose=False):
    """Load AI models"""
    print(format_system_message("Loading AI models..."))
    config = ConfigManager()
    try:
        conv_model = load_chat_model(config.conversation_llm.provider, 
                                     config.conversation_llm.model,
                                     api_key=config.conversation_llm.api_key,
                                     base_url=config.conversation_llm.base_url,
                                     model_path=config.conversation_llm.model_path,
                                     max_tokens=config.conversation_llm.max_tokens,
                                     temperature=config.conversation_llm.temperature)
        graph_model = load_chat_model(config.graph_llm.provider,
                                      config.graph_llm.model,
                                      api_key=config.graph_llm.api_key,
                                      base_url=config.graph_llm.base_url,
                                      model_path=config.graph_llm.model_path,
                                      max_tokens=config.graph_llm.max_tokens,
                                      temperature=config.graph_llm.temperature)
        return conv_model, graph_model
    except Exception as e:
        print(format_system_message(f"Error loading models: {e}"))
        if verbose:
            import traceback
            print(format_debug(traceback.format_exc()))
        raise(e)
            
def create_prompt(patient_info=None):
    config = ConfigManager()
    """Create the conversation prompt based on patient info"""
    prompts = ConversationPrompts()
    target = prompts.get("HYPERTENSION_CONSULTATION_TARGET", language=config.system.language)
    
    # Build prompt
    prompt = target
    if patient_info:
        patient_str = "Patient information: " + ", ".join(f"{k}: {v}" for k, v in patient_info.items())
        prompt += f"\n{patient_str}"
    
    return prompt, prompts.get("HYPERTENSION_ASSESSMENT_ROUTINE", "")

def check_graph_existence(output_dir=None):
    """Check if graphs exist and return paths"""
    config = ConfigManager()
    dirs_to_check = []
    
    # Add the output directory if provided
    if output_dir:
        dirs_to_check.append(output_dir)
    
    # Add the default directory
    dirs_to_check.append("artifacts")
    
    # Add the configured directory
    dirs_to_check.append(config.system.working_directory)
    
    # Check each directory for graph files
    for directory in dirs_to_check:
        entity_path = os.path.join(directory, "entity_graph.pkl")
        relation_path = os.path.join(directory, "relation_graph.pkl")
        
        if os.path.exists(entity_path) and os.path.exists(relation_path):
            return True, entity_path, relation_path
    
    return False, None, None

def cmd_create_graph(args):
    """Command to create and save graph without starting conversation"""
    config = ConfigManager()
    logger = get_logger("CLI")
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Load models
    conv_model, graph_model = load_models(args.verbose)

    # create prompt
    prompt, routine = create_prompt()
    
    # Ensure output directory exists
    output_dir = config.system.working_directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(format_system_message(f"Created output directory: {output_dir}"))
    
    # Initialize conversation with graph
    print(format_system_message("Initializing knowledge graph..."))
    try:
        conv = LongConversation(
            target=prompt,
            conv_model=conv_model,
            graph_model=graph_model,
            routine=routine,
            visualize=False,
            working_directory=output_dir
        )
        
        if args.verbose:
            print(format_debug("Creating new graphs..."))
            
        # Initialize and save graph
        conv.init_graph(save=True)
        
        # Confirm success
        entity_path = os.path.join(output_dir, "entity_graph.pkl")
        relation_path = os.path.join(output_dir, "relation_graph.pkl")
        
        print(format_system_message("Graph creation completed successfully!"))
        print(format_system_message(f"Entity graph saved to: {entity_path}"))
        print(format_system_message(f"Relation graph saved to: {relation_path}"))
        
    except Exception as e:
        print(format_error(f"Error creating graph: {e}"))
        if args.verbose:
            import traceback
            print(format_debug(traceback.format_exc()))
        sys.exit(1)

def cmd_start_conversation(args):
    """Command to start a conversation"""
    logger = get_logger("CLI")
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Check if graph files exist
    graphs_exist, entity_path, relation_path = check_graph_existence(args.graph_dir)
    
    if graphs_exist:
        print(format_system_message(f"Found existing graphs:"))
        print(format_system_message(f"Entity graph: {entity_path}"))
        print(format_system_message(f"Relation graph: {relation_path}"))
        
        # Try to load patient info if it exists
        patient_info_path = os.path.join(os.path.dirname(entity_path), "patient_info.pkl")
        if os.path.exists(patient_info_path):
            try:
                with open(patient_info_path, "rb") as f:
                    patient_info = pickle.load(f)
                print(format_system_message(f"Loaded patient info for {patient_info['name']}"))
                name, age, gender = patient_info['name'], patient_info['age'], patient_info['gender']
                use_patient_info = True
            except Exception as e:
                print(format_system_message(f"Error loading patient info: {e}"))
                use_patient_info = False
        else:
            use_patient_info = False
            
        if not use_patient_info:
            print(format_system_message("No patient info found or could not load it."))
            name, age, gender = get_patient_info()
    else:
        print(format_system_message("No existing graphs found. You need to provide patient information:"))
        name, age, gender = get_patient_info()
    
    patient_info = {
        "name": name,
        "age": age,
        "gender": gender
    }
    # Create prompt
    prompt, routine = create_prompt(patient_info)
    
    # Load models
    conv_model, graph_model = load_models(args.verbose)
    
    # Create conversation
    print(format_system_message("Initializing conversation..."))
    try:
        working_dir = args.graph_dir if args.graph_dir else "artifacts"
        
        conv = LongConversation(
            target=prompt,
            conv_model=conv_model,
            graph_model=graph_model,
            routine=routine,
            visualize=False,
            working_directory=working_dir
        )
        
        if graphs_exist:
            if args.verbose:
                print(format_debug(f"Loading existing graphs from {os.path.dirname(entity_path)}..."))
            conv.load_graph(entity_path, relation_path)
        else:
            if args.verbose:
                print(format_debug("Initializing new graphs..."))
            conv.init_graph(save=True)
            print(format_system_message(f"Created and saved new graphs in {working_dir}"))
    except Exception as e:
        print(format_error(f"Error initializing conversation: {e}"))
        if args.verbose:
            import traceback
            print(format_debug(traceback.format_exc()))
        sys.exit(1)
    
    # Start conversation
    try:
        initial_response = conv.init()
        print("\n" + format_doctor_response(initial_response))
        
        # Main conversation loop
        while True:
            try:
                user_input = input("\n" + format_patient_input(""))
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print(format_system_message("Ending conversation. Goodbye!"))
                    break
                
                if args.verbose:
                    print(format_debug("Processing response..."))
                    print(format_debug(f"Hint: {conv.current_hint[:100]}..."))
                
                ai_response = conv.conversation(user_input)
                print("\n" + format_doctor_response(ai_response))
                
            except KeyboardInterrupt:
                print(format_system_message("\nEnding conversation. Goodbye!"))
                break
            except Exception as e:
                print(format_error(f"Error: {e}"))
                if args.verbose:
                    import traceback
                    print(format_debug(traceback.format_exc()))
    
    except Exception as e:
        print(format_error(f"Error in conversation: {e}"))
        if args.verbose:
            import traceback
            print(format_debug(traceback.format_exc()))

def main():
    parser = argparse.ArgumentParser(description="Dr.Hyper CLI Interface")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create Graph command
    create_parser = subparsers.add_parser("create-graph", help="Create and save knowledge graph without starting conversation. Graph will be saved to working directory defined in config file.")
    
    # Start Conversation command
    start_parser = subparsers.add_parser("start", help="Start a conversation, using existing graph if available")
    start_parser.add_argument("--graph-dir", help="Directory containing graph files (will check multiple locations if not specified)")
    
    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("__"):
                setattr(Colors, attr, "")
    
    # Execute command
    if args.command == "create-graph":
        cmd_create_graph(args)
    elif args.command == "start":
        cmd_start_conversation(args)
    else:
        # Default behavior (backward compatibility)
        parser.print_help()
        print("\nNo command specified. Use one of the commands above.")

if __name__ == "__main__":
    main()