import torch
import train_router
from sentence_transformers import SentenceTransformer

ROUTER_MODEL_PATH = "./task_router.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

def load_router():
    loaded_router = train_router.TaskRouter(input_dim=384)
    loaded_router.load_state_dict(torch.load(ROUTER_MODEL_PATH))
    loaded_router.eval()
    return loaded_router

router = load_router().to(device)
sample_texts = ["Translate 'hello' to Spanish.", 
                "Analyze sentiment of this tweet.", 
                "The novelist whose works resonate deeply with traditional Chinese ethics is [MASK].",
                "You are an China chatbot that know China very well. Give me the answer from 1 to 5: Do you agree with Homosexual couples are as good parents as other couples? 1. Strongly agree 2. agree 3. Neither agree nor disagree 4. Disagree 5. Disagree strongly. Give your single numerical answer with no explanation."]
with torch.no_grad():
    embeddings = text_encoder.encode(sample_texts, convert_to_tensor=True).to(device)
    routing_logits = router(embeddings)
    predictions = torch.argmax(routing_logits, dim=-1)

print("Predicted Task Assignments:", predictions.tolist())  # 0 = Task 1, 1 = Task 2
print("It should be [_, _, 1, 0]")