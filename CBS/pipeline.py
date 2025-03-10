import SFT
import DPO
import CBS
import config
import utils

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    TrainingArguments,
)


N_EPOCHS = [1]
SFT_LR = [1e-05, 5e-05, 2e-04, 1e-03]
DPO_LR = [2e-07, 1e-06, 4e-06, 2e-05]

def main():
    # assert(len(N_EPOCHS) == len(SFT_LR))
    # assert(len(SFT_LR) == len(DPO_N_EPOCHS))
    # assert(len(SFT_LR) == len(DPO_LR))
    assert(len(SFT_LR) == len(DPO_LR))

    for i in range(len(N_EPOCHS)):
        for j in range(len(SFT_LR)):
            # lora_r = LORA_R_LIST[i]
            # lora_alpha = LORA_ALPHA_LIST[i]
            # lora_dropout = LORA_DROPOUT_LIST[i]
            # print(f"Got LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")

            sft_n_epochs = N_EPOCHS[i]
            dpo_n_epochs = N_EPOCHS[i]
            sft_lr = SFT_LR[j]
            dpo_lr = DPO_LR[j]
            print(f"Got sft_n_epochs: {sft_n_epochs}, sft_lr: {sft_lr}, dpo_n_epochs: {dpo_n_epochs}, dpo_lr: {dpo_lr}")

            if sft_n_epochs == 3 and sft_lr == 5e-05:
                continue

            # create args and modify contents
            args = config.Args()


            args.model_name = f"sft-dpo_epochs_lr_{sft_n_epochs}_{sft_lr}_{dpo_lr}"
            args.sft_adapter_save_path = f"finetuned_models/sft-epochs_lr_{sft_n_epochs}_{sft_lr}_{dpo_lr}"
            args.dpo_adapter_save_path = f"finetuned_models/{args.model_name}"

            # args.lora_r = lora_r
            # args.lora_alpha = lora_alpha
            # args.lora_dropout = lora_dropout
            args.sft_n_epochs = sft_n_epochs
            args.sft_lr = sft_lr
            args.dpo_n_epochs = dpo_n_epochs
            args.dpo_lr = dpo_lr

            sft_config = config.SFTConfig(args)
            dpo_config = config.DPOConfig(args)
            cbs_config = config.CBSConfig(args)

            sft_model, sft_tokenizer = SFT.get_sft_model_and_tokenizer(sft_config)

            DPO.dpo_finetune(dpo_config, sft_model, sft_tokenizer)

            CBS.cbs(cbs_config)

if __name__ == "__main__":
    main()