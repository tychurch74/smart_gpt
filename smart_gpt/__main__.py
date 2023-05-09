"""Entry point for smart_gpt implementation."""

import datetime
import json
from semantic_search import SemanticSearch
from chain_of_thought import chain_of_thought_response


def perform_semantic_search(message_history, input_string):
    """Perform semantic search on given message history with the input string."""
    semantic_search = SemanticSearch(message_history)
    related_content = semantic_search.semantic_search(input_string)
    return related_content


def main():
    """Main function.

    Change the initial_guidance and eval_metrics variables to change the type of response.
    Date and time are printed to the console to show how long the response takes to generate, purely for testing purposes.
    Refer to gpt_agents.py for more information about the chain of prompts used.
    """
    long_term_memory = True
    initial_guidance = "factual, logical, and clear"
    eval_metrics = "factual accuracy, logical consistency, and clarity"

    user_input = input("Enter your question: ")

    if long_term_memory:
        with open("data/full_message_history.json", "r") as f:
            previous_messages = json.load(f)
        full_message_history = previous_messages
        joined_full_message_history = " ".join(
            [message["content"] for message in full_message_history]
        )
        context = perform_semantic_search(joined_full_message_history, user_input)
    else:
        context = "none"

    print(f"Using {context} as context\n\n")
    print("Start time: ")
    print(datetime.datetime.now())
    print("\n\n")

    response = chain_of_thought_response(user_input, initial_guidance, context, eval_metrics)

    print(f"Final response: {response}\n\n")
    print("End time: ")
    print(datetime.datetime.now())


if __name__ == "__main__":
    main()

