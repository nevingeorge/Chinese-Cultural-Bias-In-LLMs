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


def main():
    args = config.Args()

    args.model_name = f"cbs_sft-dpo"
    args.sft_adapter_save_path = f"finetuned_models/csb_sft"
    args.dpo_adapter_save_path = f"finetuned_models/{args.model_name}"
    args.cbs_adapter_path = f"finetuned_models/{args.model_name}"

    sft_config = config.SFTConfig(args)
    dpo_config = config.DPOConfig(args)
    cbs_config = config.CBSConfig(args)

    sft_model, sft_tokenizer = SFT.get_sft_model_and_tokenizer(sft_config)

    DPO.dpo_finetune(dpo_config, sft_model, sft_tokenizer)

    CBS.cbs(cbs_config)

if __name__ == "__main__":
    main()