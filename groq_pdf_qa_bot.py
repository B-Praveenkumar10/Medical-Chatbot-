import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

# Initialize Groq client
groq_api_key = "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"
groq_client = Groq(api_key=groq_api_key)

# Initialize SentenceTransformer for creating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone_api_key = "201e8335-53fe-4252-8534-f8f2cc116510"
pc = Pinecone(api_key=pinecone_api_key)

index_name = "pdf-qa-index"
index = pc.Index(index_name)


def find_most_relevant(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [result['metadata']['text'] for result in results['matches']]


def answer_question(question, context):
    prompt = f"""Context: {context}

Question: {question}

Answer based on the provided context and, if necessary, include additional relevant information:"""

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that provides accurate answers by combining information from the provided context with additional knowledge if needed."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Streamlit interface
st.title("PDF Question Answering Bot")

# Initialize session state to store the conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User input
question = st.text_input("Ask your question:")

if st.button("Send") and question:
    # Retrieve relevant context from the database
    relevant_texts = find_most_relevant(question)
    context = " ".join(relevant_texts)

    # Get the answer from the model
    answer = answer_question(question, context)

    # Append the question and answer to the conversation
    st.session_state.conversation.append({"role": "user", "content": question})
    st.session_state.conversation.append({"role": "assistant", "content": answer})

# Display the conversation
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

