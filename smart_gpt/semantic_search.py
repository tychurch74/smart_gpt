import sqlite3
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_similarities(past_vectors, user_vector):
    # Calculate cosine similarities between the user's input and past sentences
    return cosine_similarity([user_vector], past_vectors)[0]


def get_most_similar(similarities, k):
    # Get the indices of the k most similar past sentences
    most_similar_indices = np.argsort(similarities)[-k:]

    # Return the most similar sentences in ascending order of similarity
    return most_similar_indices[::-1]


def embed_sentences(sentences):
    # Embed a list of sentences
    return [embed_sentence(sentence) for sentence in sentences]


def embed_sentence(sentence):
    # Embed a single sentence
    response = openai.Embedding.create(
        input=sentence,
        model="text-embedding-ada-002"
    )
    # Get the embedding vector from the response
    return response["data"][0]["embedding"]


def save_conversation_history(user_input, chatbot_response):
    # Connect to the SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect('chat_history.db')

    # Create a cursor object
    cur = conn.cursor()

    # Create table if it doesn't exist
    cur.execute('''CREATE TABLE IF NOT EXISTS chat_history
                (user_input TEXT, chatbot_response TEXT)''')

    # Insert a row of data
    cur.execute(f"INSERT INTO chat_history VALUES (?, ?)",
                (user_input, chatbot_response))

    # Save (commit) the changes and close the connection
    conn.commit()
    conn.close()


def get_past_conversations():
    # Connect to the SQLite database
    conn = sqlite3.connect('chat_history.db')
    cur = conn.cursor()

    # Get all past conversations
    cur.execute("SELECT * FROM chat_history")
    past_conversations = cur.fetchall()

    # Close the connection
    conn.close()

    # Return past conversations as a list of tuples
    return past_conversations


def semantic_search(past_conversations, user_input):
    # Extract the sentences from the past conversations
    past_sentences = [conv[0] for conv in past_conversations]

    # Convert conversations and user input to vectors
    past_vectors = embed_sentences(past_sentences)
    user_vector = embed_sentence(user_input)

    # Calculate similarities
    similarities = calculate_similarities(past_vectors, user_vector)

    # Get the 5 most similar past conversations
    most_similar_indices = get_most_similar(similarities, 5)

    # Get the most similar sentences
    most_similar_sentences = [past_sentences[i] for i in most_similar_indices]

    return most_similar_sentences