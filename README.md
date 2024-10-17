# Medical Chatbot using Groq and Pinecone

## Overview

This project implements a medical chatbot that leverages Groq for natural language processing and Pinecone for vector similarity search. The chatbot is designed to answer medical questions based on the content extracted from PDF documents, providing users with accurate, evidence-based medical information.

## Features

- **PDF Processing**: Extracts text from medical PDF documents for question-answering.
- **Semantic Search**: Utilizes embeddings to find the most relevant chunks of text based on user queries.
- **Medical Contextual Understanding**: Provides answers based on the context of the medical documents, ensuring evidence-based responses.
- **Robust Error Handling**: Handles errors gracefully and provides informative feedback to users.

## Technologies Used

- **Python**: The primary programming language for the implementation.
- **Chainlit**: Framework for creating interactive chat applications.
- **Sentence Transformers**: For creating semantic embeddings from text.
- **Pinecone**: Managed vector database for efficient similarity search.
- **Groq**: AI model for generating responses based on input prompts.
- **PyPDF2**: Library for extracting text from PDF files.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required libraries:
  - `chainlit`
  - `sentence-transformers`
  - `pinecone`
  - `groq`
  - `PyPDF2`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-chatbot.git
   cd medical-chatbot
   ```

2. Install the required libraries:
   ```bash
   pip install chainlit sentence-transformers pinecone-client groq PyPDF2
   ```

3. Set up your API keys:
   Replace `YOUR_GROQ_API_KEY` and `YOUR_PINECONE_API_KEY` in the code with your actual API keys.

## Usage

1. Place your medical PDF documents in the `data` directory.
2. Start the chatbot:
   ```bash
   chainlit run medical_chatbot.py
   ```
3. Open your browser and navigate to the provided URL to interact with the chatbot.

## Example Interaction

User: What are the symptoms of diabetes?

Chatbot: The symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, and blurred vision, among others. If you have concerns about diabetes, consult a healthcare professional.

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
