from transformers import pipeline

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Context and question
context = """
Hawking radiation is a theoretical prediction that black holes emit radiation due to quantum effects near the event horizon.
This radiation is named after physicist Stephen Hawking, who first proposed the idea in 1974.
Black holes are regions of space where gravity is so strong that nothing, not even light, can escape.
"""

question = "Who proposed the idea of Hawking radiation?"

# Get the answer
result = qa_pipeline(question=question, context=context)

print(f"Answer: {result['answer']}")