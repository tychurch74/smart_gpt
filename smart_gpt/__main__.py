import os
import openai

from semantic_search import save_conversation_history, semantic_search, get_past_conversations
from chain_of_thought import chain_of_thought_response



def chatbot_response(user_input, context, initial_guidance):
    system_message = {"role": "system", "content": f"You are an AI language model trained by OpenAI to assist users with answering various questions or requests. Your responses should be {initial_guidance}. You may use the following information from past conversations with the user (if there have been any) to help with your responses: {context}"}
    user_message = {"role": "user", "content": user_input}
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message],
        temperature=0.5,
    )

    return response["choices"][0]["message"]["content"] # returns chatbot response as a string


def no_past_conversations(chain_of_thought=False):
    initial_guidance = "factual, logical, and clear"
    eval_metrics = "factual accuracy, logical consistency, and clarity"
    user_input = "user said: " + input("Enter your question: ")
    context = "no conversation history yet"
    if chain_of_thought:
        chatbot_msg = "assistant said: " + chain_of_thought_response(user_input, initial_guidance, context, eval_metrics)
    else:
        chatbot_msg = "assistant said: " + chatbot_response(user_input, context, initial_guidance)
    print(chatbot_msg)
    save_conversation_history(user_input, chatbot_msg)


def has_past_conversations(chain_of_thought=False):
    initial_guidance = "factual, logical, and clear"
    eval_metrics = "factual accuracy, logical consistency, and clarity"
    past_conversations = get_past_conversations()
    user_input = "user said: " + input("Enter your question: ")
    context = " ".join(semantic_search(past_conversations, user_input))
    if chain_of_thought:
        chatbot_msg = "assistant said: " + chain_of_thought_response(user_input, initial_guidance, context, eval_metrics)
    else:
        chatbot_msg = "assistant said: " + chatbot_response(user_input, context, initial_guidance)
    print(chatbot_msg)
    save_conversation_history(user_input, chatbot_msg)


def main():
    while True:
        use_chain_of_thought = input("Use chain of thought? (y/n): ")
        if use_chain_of_thought == 'y':
            chain_of_thought = True
        else:
            chain_of_thought = False
        if os.path.exists('chat_history.db'):
            has_past_conversations(chain_of_thought=chain_of_thought)
        else:
            no_past_conversations(chain_of_thought=chain_of_thought)


if __name__ == "__main__":
    main()

