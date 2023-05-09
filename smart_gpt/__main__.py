"""Entry point for smart_gpt implementation."""

import datetime
from chain_of_thought import chain_of_thought_response


def main():
    """Main function.
    
    Change the initial_guidance and eval_metrics variables to change the type of response.
    
    Date and time are printed to the console to show how long the response takes to generate, purely for testing purposes.

    Refer to gpt_agents.py for more information about the chain of prompts used.
    
    """
    initial_guidance = "factual, logical, and clear"
    eval_metrics = "factual accuracy, logical consistency, and clarity"
   
    user_input = input("Enter your question: ")
    
    print("Start time: ")
    print(datetime.datetime.now())
    print("\n\n")
    
    response = chain_of_thought_response(user_input, initial_guidance, eval_metrics)
    
    print(f"final response: {response}\n\n")
    print("End time: ")
    print(datetime.datetime.now())


if __name__ == "__main__":  
    main()
