---
library_name: peft
license: other
base_model: /home/hf/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/
tags:
- base_model:adapter:/home/hf/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: conll12_en_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# conll12_en_lora

This model is a fine-tuned version of [/home/hf/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/](https://huggingface.co//home/hf/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/) on the srl_format_pred_arg_token_v2 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- lr_scheduler_warmup_steps: 100
- training_steps: 200000
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 4.47.1
- Pytorch 2.11.0+cu128
- Datasets 4.8.4
- Tokenizers 0.21.0