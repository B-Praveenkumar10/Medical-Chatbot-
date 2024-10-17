import chainlit as cl

# Define a simple conversational bot logic
def simple_bot(user_input):
    if "hello" in user_input.lower():
        return "Hello! How can I assist you today?"
    elif "bye" in user_input.lower():
        return "Goodbye! Have a great day!"
    else:
        return "I'm not sure how to respond to that. Can you please rephrase?"

# Define a Chainlit handler
@cl.on_message
async def main(msg):
    user_input = msg['content']
    response = simple_bot(user_input)
    await cl.send_message(response)

if __name__ == "__main__":
    cl.run(main)
