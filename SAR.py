import torch
import torch.nn.functional as F
from trl import SFTTrainer

class SARTrainer(SFTTrainer):
    def __init__(self, *args, epsilon=1e-3, alpha=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        
        outputs = model(**inputs)
        task_loss = outputs.loss
        clean_logits = outputs.logits
        
        if self.alpha > 0:
            embedding_layer = model.get_input_embeddings()
            
            with torch.no_grad():
                orig_embeds = embedding_layer(input_ids)
            
            inputs_embeds = orig_embeds.clone().detach().requires_grad_(True)
            
            perturbed_inputs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            adv_outputs = model(**perturbed_inputs)
            adv_loss = adv_outputs.loss
            
            gradients = torch.autograd.grad(
                outputs=adv_loss,
                inputs=inputs_embeds,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            perturbation = self.epsilon * gradients.sign()
            
            perturbed_embeds = inputs_embeds + perturbation
            
            perturbed_outputs = model(
                inputs_embeds=perturbed_embeds,
                attention_mask=attention_mask,
                input_ids=None
            )
            perturbed_logits = perturbed_outputs.logits
            
            sar_loss = F.kl_div(
                F.log_softmax(clean_logits.detach(), dim=-1), 
                F.softmax(perturbed_logits, dim=-1), 
                reduction="batchmean"
            )
            
            total_loss = task_loss + self.alpha * sar_loss
        else:
            total_loss = task_loss
        
        return (total_loss, outputs) if return_outputs else total_loss