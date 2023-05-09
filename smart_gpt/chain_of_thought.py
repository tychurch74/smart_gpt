import openai
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# Function to generate chatbot response
def generate_chatbot_response(
    system_message_content, user_message_content, model_temperature=0.5
):
    system_message = {"role": "system", "content": system_message_content}
    user_message = {"role": "user", "content": user_message_content}

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message],
        temperature=model_temperature,
    )

    return response["choices"][0]["message"]["content"]

def chain_of_thought_response(user_input, initial_guidance, eval_metrics):
    # Generate response options
    ai_system_msg = f"You are an AI language model trained by OpenAI to assist users with answering various questions or requests. Your responses should be {initial_guidance}. If you are unsure about some of the information in your response, preface that information by saying you are not totally sure."
    preface_message = f"Question: {user_input} Answer: Let's work this out in a step by step way to be sure we have the right answer."

    with ThreadPoolExecutor() as executor:
        # Submit the API calls concurrently
        futures = [executor.submit(generate_chatbot_response, ai_system_msg, preface_message) for _ in range(3)]

        # Collect the results as they complete
        response_options = [future.result() for future in as_completed(futures)]

    formatted_options = ", ".join(
        f"Option {i + 1}: {response_options[i]}" for i in range(3)
    )
    
    # Generate researcher evaluation
    ai_researcher_msg = f"You are an AI language model trained by OpenAI to act as a researcher tasked with evaluating the quality of responses to a user's prompt based on the following metrics: {eval_metrics}. Be sure to include your reasoning for each of your evaluations of the given responses."
    researcher_prompt = f"Original prompt: {user_input} Response options: {formatted_options}. You are a researcher tasked with evaluating the quality of these response options. List any flaws or faulty logic of each response. Let's think about this step by step:"
    researcher_evaluation = generate_chatbot_response(
        ai_researcher_msg, researcher_prompt
    )

    # Find the best response and improve it
    ai_decision_msg = f"You are an AI language model trained by OpenAI to decide which response option a research evaluator thought was best and then improve the quality of the chosen response based on the following metrics: {eval_metrics} and finally print out your final improved response for the user to see."
    final_response_prompt = f"Find which of the following responses the researcher thought was best, and improve that response to be used as a final response to a user's question or request. original responses: {formatted_options} researcher evalutions: {researcher_evaluation}"
    final_response = generate_chatbot_response(ai_decision_msg, final_response_prompt)

    return final_response

