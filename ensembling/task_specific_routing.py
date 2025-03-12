import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer
import train_router

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FINAL_MODEL_PATH = "../models/task_routing_model"
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

def load_model(model_path):
    """Load either a Hugging Face model or a local PEFT adapter model."""

    peft_config = PeftConfig.from_pretrained(model_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
    model = model.merge_and_unload()
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return pipe

def test_task_routing_transformer(model, sample_text):
    with torch.no_grad():
        return model(sample_text)
    
def generate_output(pipe1, pipe2, router, text_input):
    text_embedding = text_encoder.encode(text_input, convert_to_tensor=True).unsqueeze(0).to(device)

    with torch.no_grad():
        routing_logits = router(text_embedding)
        task_assignment = torch.argmax(routing_logits, dim=-1).item()

    if task_assignment == 0:
        print("Task 0")
        return pipe1(text_input, max_new_tokens=10, pad_token_id=128001)
    else:
        print("Task 1")
        return pipe2(text_input)

def main():
    pipe1 = load_model("../models/SFT-LoRA-Llama-3.2-1B-Instruct")
    pipe2 = load_model("../models/sft-dpo_epochs_lr_6_0.0002_4e-06")

    # Load trained router model 
    router = train_router.TaskRouter(input_dim=384).to(device)
    router.load_state_dict(torch.load("./task_router.pth"))
    router.eval()

    print("Generated Output:", generate_output(pipe1, pipe2, router, "Translate 'hello' to Spanish."))

if __name__ == "__main__":
    main()