---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf
model-index:
- name: desc_full_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# desc_full_lora

This model is a fine-tuned version of [/bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf](https://huggingface.co//bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf) on the ec_desc_train_dataset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1223

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 3
- eval_batch_size: 3
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- total_train_batch_size: 12
- total_eval_batch_size: 12
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- num_epochs: 5.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.1448        | 0.13  | 100  | 0.1444          |
| 0.1399        | 0.25  | 200  | 0.1376          |
| 0.1403        | 0.38  | 300  | 0.1356          |
| 0.1325        | 0.51  | 400  | 0.1355          |
| 0.1334        | 0.63  | 500  | 0.1337          |
| 0.133         | 0.76  | 600  | 0.1326          |
| 0.1318        | 0.88  | 700  | 0.1309          |
| 0.1283        | 1.01  | 800  | 0.1304          |
| 0.1281        | 1.14  | 900  | 0.1298          |
| 0.1279        | 1.26  | 1000 | 0.1284          |
| 0.1297        | 1.39  | 1100 | 0.1272          |
| 0.1257        | 1.52  | 1200 | 0.1279          |
| 0.1184        | 1.64  | 1300 | 0.1267          |
| 0.1249        | 1.77  | 1400 | 0.1248          |
| 0.1155        | 1.89  | 1500 | 0.1256          |
| 0.1229        | 2.02  | 1600 | 0.1238          |
| 0.1205        | 2.15  | 1700 | 0.1244          |
| 0.1181        | 2.27  | 1800 | 0.1242          |
| 0.1172        | 2.4   | 1900 | 0.1231          |
| 0.1173        | 2.53  | 2000 | 0.1225          |
| 0.1168        | 2.65  | 2100 | 0.1220          |
| 0.1148        | 2.78  | 2200 | 0.1215          |
| 0.1156        | 2.9   | 2300 | 0.1206          |
| 0.1136        | 3.03  | 2400 | 0.1228          |
| 0.106         | 3.16  | 2500 | 0.1233          |
| 0.1089        | 3.28  | 2600 | 0.1223          |
| 0.1011        | 3.41  | 2700 | 0.1226          |
| 0.1006        | 3.54  | 2800 | 0.1232          |
| 0.1044        | 3.66  | 2900 | 0.1221          |
| 0.1057        | 3.79  | 3000 | 0.1223          |
| 0.1022        | 3.91  | 3100 | 0.1212          |
| 0.0954        | 4.04  | 3200 | 0.1312          |
| 0.0888        | 4.17  | 3300 | 0.1316          |
| 0.0892        | 4.29  | 3400 | 0.1312          |
| 0.0869        | 4.42  | 3500 | 0.1312          |
| 0.0841        | 4.55  | 3600 | 0.1313          |
| 0.0899        | 4.67  | 3700 | 0.1313          |
| 0.0863        | 4.8   | 3800 | 0.1313          |
| 0.0849        | 4.92  | 3900 | 0.1313          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2