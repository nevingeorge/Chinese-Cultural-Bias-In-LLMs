import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, pipeline
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
    
class TaskRoutingTransformer(PreTrainedModel):
    def __init__(self, config, router, task1_model, task2_model):
        super().__init__(config)
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
            return self.task1_model(text_input)
        else:
            return self.task2_model(text_input)

def test_task_routing_transformer(model, sample_text):
    with torch.no_grad():
        return model(sample_text)

def main():
    # Load pre-trained models for each task
    task1_model = TaskModel("../models/SFT-LoRA-Llama-3.2-1B-Instruct").to(device)
    task2_model = TaskModel("../models/sft-dpo_epochs_lr_6_0.0002_4e-06").to(device)

    # Load trained router model 
    router = train_router.TaskRouter(input_dim=384)
    router.load_state_dict(torch.load("./task_router.pth"))
    router.eval()

    final_model = TaskRoutingTransformer(task1_model.model.config, router, task1_model, task2_model)

    print("Generated Output:", test_task_routing_transformer(final_model, "Translate 'hello' to Spanish."))

if __name__ == "__main__":
    main()