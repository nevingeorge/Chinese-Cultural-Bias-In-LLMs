import torch
import train_router
from sentence_transformers import SentenceTransformer

ROUTER_MODEL_PATH = "./task_router.pth"
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

def load_router():
    loaded_router = train_router.TaskRouter(input_dim=384)
    loaded_router.load_state_dict(torch.load(ROUTER_MODEL_PATH))
    loaded_router.eval()
    return loaded_router

router = load_router()
sample_texts = ["Translate 'hello' to Spanish.", "Analyze sentiment of this tweet."]
with torch.no_grad():
    embeddings = text_encoder.encode(sample_texts, convert_to_tensor=True)
    routing_logits = router(embeddings)
    predictions = torch.argmax(routing_logits, dim=-1)

print("Predicted Task Assignments:", predictions.tolist())  # 0 = Task 1, 1 = Task 2