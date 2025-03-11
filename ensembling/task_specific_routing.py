import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import json

# Create a directory to save the new model
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "task_routing_model.pth")

# Load a text embedding model
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Define Task-Specific Models (Assuming They Are Already Trained & Saved)
class TaskModel(nn.Module):
    def __init__(self, model_path):
        super(TaskModel, self).__init__()
        self.model = torch.load(model_path)  # Load model from file
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Load pre-trained models for each task
task1_model = TaskModel("path/to/task1_model.pth")
task2_model = TaskModel("path/to/task2_model.pth")

# Define Task Router (Binary Classifier)
class TaskRouter(nn.Module):
    def __init__(self, input_dim):
        super(TaskRouter, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Output logits for Task 1 (0) and Task 2 (1)

    def forward(self, x):
        return self.fc(x)

# Load trained router model (assumes it's already trained)
router = TaskRouter(input_dim=384)
router.load_state_dict(torch.load("saved_models/task_router.pth"))
router.eval()  # Set to evaluation mode

# Define New Model That Handles Task-Specific Routing & Processing
class TaskRoutingModel(nn.Module):
    def __init__(self, router, task1_model, task2_model):
        super(TaskRoutingModel, self).__init__()
        self.router = router
        self.task1_model = task1_model
        self.task2_model = task2_model

    def forward(self, text_input):
        # Convert text to embedding
        text_embedding = text_encoder.encode(text_input, convert_to_tensor=True).unsqueeze(0)

        # Use router to determine the task
        with torch.no_grad():
            routing_logits = self.router(text_embedding)
            task_assignment = torch.argmax(routing_logits, dim=-1).item()

        # Route to the correct model
        if task_assignment == 0:
            output = self.task1_model(text_embedding)
        else:
            output = self.task2_model(text_embedding)

        return output

# Instantiate the final routing model
task_routing_model = TaskRoutingModel(router, task1_model, task2_model)

# Save the new model
torch.save(task_routing_model.state_dict(), FINAL_MODEL_PATH)
print(f"Final routing model saved to {FINAL_MODEL_PATH}")

# Function to Load and Use the Final Routing Model
def load_task_routing_model():
    loaded_model = TaskRoutingModel(router, task1_model, task2_model)
    loaded_model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    loaded_model.eval()
    return loaded_model

# Load and Test the Final Routing Model
task_routing_model = load_task_routing_model()
sample_text = "Translate 'hello' to Spanish."

with torch.no_grad():
    output = task_routing_model(sample_text)

print("Generated Output:", output)