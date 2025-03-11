import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer
import train_router

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FINAL_MODEL_PATH = "../models/task_routing_model"
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

class TaskModel(nn.Module):
    def __init__(self, model_path):
        super(TaskModel, self).__init__()
        peft_config = PeftConfig.from_pretrained(model_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
        model = model.merge_and_unload()
        model.eval()

        self.model = model

    def forward(self, x):
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        return pipe(x)

# Load pre-trained models for each task
task1_model = TaskModel("../models/SFT-LoRA-Llama-3.2-1B-Instruct").to(device)
task2_model = TaskModel("../models/sft-dpo_epochs_lr_6_0.0002_4e-06").to(device)

# Load trained router model 
router = train_router.TaskRouter(input_dim=384)
router.load_state_dict(torch.load("./task_router.pth"))
router.eval()

class TaskRoutingModel(nn.Module):
    def __init__(self, router, task1_model, task2_model):
        super(TaskRoutingModel, self).__init__()
        self.router = router
        self.task1_model = task1_model
        self.task2_model = task2_model

    def forward(self, text_input):
        text_embedding = text_encoder.encode(text_input, convert_to_tensor=True).unsqueeze(0).to(device)

        # Use router to determine the task
        with torch.no_grad():
            routing_logits = self.router(text_embedding)
            task_assignment = torch.argmax(routing_logits, dim=-1).item()

        # Route to the correct model
        if task_assignment == 0:
            output = self.task1_model(text_input)
        else:
            output = self.task2_model(text_input)

        return output

task_routing_model = TaskRoutingModel(router, task1_model, task2_model).to(device)

torch.save(task_routing_model.state_dict(), FINAL_MODEL_PATH)
print(f"Final routing model saved to {FINAL_MODEL_PATH}")

# # Function to Load and Use the Final Routing Model
# def load_task_routing_model():
#     loaded_model = TaskRoutingModel(router, task1_model, task2_model)
#     loaded_model.load_state_dict(torch.load(FINAL_MODEL_PATH))
#     loaded_model.eval()
#     return loaded_model

# # Load and Test the Final Routing Model
# task_routing_model = load_task_routing_model()
# sample_text = "Translate 'hello' to Spanish."

# with torch.no_grad():
#     output = task_routing_model(sample_text)

# print("Generated Output:", output)