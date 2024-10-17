import streamlit as st
import random


def get_bot_response(user_input):
    # This is a simple mock-up of a response. In a real application,
    # you would integrate with an actual language model API.
    responses = [
        "That's an interesting point. Can you tell me more?",
        "I understand. Have you considered looking at it from a different perspective?",
        "That's a complex topic. Let me think about that for a moment...",
        "I'm not sure I have all the information to answer that fully. Could you provide more context?",
        "That's a great question! Here's what I think..."
    ]
    return random.choice(responses)


st.title("ChatGPT-like Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_bot_response(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})