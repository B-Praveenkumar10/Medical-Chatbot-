import chainlit as cl
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
import PyPDF2
import os

# Initialize Groq client
groq_api_key = "YOUR_GROQ_API_KEY"  # Replace with your Groq API key
groq_client = Groq(api_key=groq_api_key)

# Initialize SentenceTransformer for creating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone_api_key = "YOUR_PINECONE_API_KEY"  # Replace with your Pinecone API key
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-pdf-qa-index"
index = pc.Index(index_name)


def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        end = start + chunk_size
    return chunks


def index_pdf(pdf_path):
    text = process_pdf(pdf_path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode([chunk]).tolist()[0]
        index.upsert(vectors=[{
            'id': f'chunk_{i}',
            'values': embedding,
            'metadata': {'text': chunk}
        }])


def find_most_relevant(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [result['metadata']['text'] for result in results['matches']]


def answer_question(question, context):
    prompt = f"""Context: {context}

Question: {question}

Answer based on the provided medical context, utilizing any necessary medical guidelines and additional knowledge if needed:"""

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "system",
                       "content": "You are a knowledgeable medical assistant. Provide accurate, evidence-based medical answers based on the context given."},
                      {"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while processing the request: {str(e)}"


@cl.on_chat_start
async def start():
    # Index PDFs in a specified directory
    pdf_directory = "data"
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            index_pdf(pdf_path)


@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    relevant_texts = find_most_relevant(user_input)
    context = " ".join(relevant_texts)

    if context:
        answer = answer_question(user_input, context)
    else:
        answer = "I'm sorry, I couldn't find any relevant information for your query."

    await cl.Message(content=answer).send()
