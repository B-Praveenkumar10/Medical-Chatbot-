from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/your/llama/model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the chatbot
prompt = "Hello, how can I help you today?"
response = generate_response(prompt)
print(response)
