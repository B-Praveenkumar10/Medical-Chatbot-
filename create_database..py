import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, PodSpec
import uuid

# Initialize SentenceTransformer for creating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone_api_key = "201e8335-53fe-4252-8534-f8f2cc116510"
pc = Pinecone(api_key=pinecone_api_key)

index_name = "pdf-qa-index"

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_model.get_sentence_embedding_dimension(),
        metric='cosine',
        spec=PodSpec(
            environment="gcp-starter"
        )
    )

# Get the index
index = pc.Index(index_name)

def load_and_process_pdfs(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                yield f"{filename}-page{page_num + 1}", text

def create_and_upsert_embeddings(pdf_generator):
    batch_size = 100
    batch = []

    for doc_id, text in pdf_generator:
        embedding = embedding_model.encode(text).tolist()
        batch.append((str(uuid.uuid4()), embedding, {"text": text, "doc_id": doc_id}))

        if len(batch) == batch_size:
            index.upsert(vectors=batch)
            batch = []

    if batch:
        index.upsert(vectors=batch)

if __name__ == "__main__":
    pdf_directory = "data"  # Replace with your PDF directory path
    pdf_generator = load_and_process_pdfs(pdf_directory)
    create_and_upsert_embeddings(pdf_generator)
    print("PDF processing and embedding creation complete.")
