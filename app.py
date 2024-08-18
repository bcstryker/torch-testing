from transformers import pipeline
import torch

# Specify the model and revision explicitly
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Check if MPS (Apple's Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple's Metal Performance Shaders) for computation.")
else:
    device = torch.device("cpu")
    print("MPS not available. Using CPU for computation.")

# Initialize the pipeline with the specified model and device
classifier = pipeline("sentiment-analysis", model=model_name, device=device)

# Test the classifier with a sample sentence
result = classifier("This is fantastic!")
print(result)
