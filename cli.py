# cli.py
import argparse
import os
import sys

from core.conversation import LongConversation
from prompts.templates import ConversationPrompts
from utils.logging import get_logger
from config.settings import ConfigManager
from utils.aux import *

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
        log_messages = conv.init_graph(save=True)
        
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
    config = ConfigManager()
    logger = get_logger("CLI")
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Check if graph files exist
    graphs_exist, entity_path, relation_path = check_graph_existence(args.graph_dir)
    
    if graphs_exist:
        print(format_system_message(f"Found existing graphs:"))
        print(format_system_message(f"Entity graph: {entity_path}"))
        print(format_system_message(f"Relation graph: {relation_path}"))
        
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
        working_dir = config.system.working_directory
        
        conv = LongConversation(
            target=prompt,
            conv_model=conv_model,
            graph_model=graph_model,
            routine=routine,
            visualize=False,
            working_directory=working_dir,
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
        response_content, _ = conv.init()
        print("\n" + format_doctor_response(response_content))
        
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
                
                ai_response, is_accomplished, _ = conv.conversation(user_input)
                print(ai_response, is_accomplished)
                
                print("\n" + format_doctor_response(ai_response))
                
                if is_accomplished:
                    print(format_system_message("\nThe consultation goals have been accomplished!"))
                    print(format_system_message("Ending conversation. Goodbye!"))
                    break
                
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