---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/
model-index:
- name: only_structure
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# only_structure

This model is a fine-tuned version of [/bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/](https://huggingface.co//bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/) on the only_structure_desc_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1345

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
| 0.1598        | 0.13  | 100  | 0.1454          |
| 0.1429        | 0.25  | 200  | 0.1399          |
| 0.1396        | 0.38  | 300  | 0.1365          |
| 0.1332        | 0.51  | 400  | 0.1343          |
| 0.1353        | 0.63  | 500  | 0.1351          |
| 0.1347        | 0.76  | 600  | 0.1313          |
| 0.1314        | 0.88  | 700  | 0.1299          |
| 0.1277        | 1.01  | 800  | 0.1286          |
| 0.1252        | 1.14  | 900  | 0.1285          |
| 0.1276        | 1.26  | 1000 | 0.1283          |
| 0.1273        | 1.39  | 1100 | 0.1278          |
| 0.1264        | 1.52  | 1200 | 0.1269          |
| 0.1232        | 1.64  | 1300 | 0.1254          |
| 0.1224        | 1.77  | 1400 | 0.1248          |
| 0.1195        | 1.89  | 1500 | 0.1243          |
| 0.1234        | 2.02  | 1600 | 0.1228          |
| 0.1189        | 2.15  | 1700 | 0.1240          |
| 0.1172        | 2.27  | 1800 | 0.1237          |
| 0.1139        | 2.4   | 1900 | 0.1228          |
| 0.114         | 2.53  | 2000 | 0.1221          |
| 0.1152        | 2.65  | 2100 | 0.1225          |
| 0.1127        | 2.78  | 2200 | 0.1211          |
| 0.1119        | 2.9   | 2300 | 0.1207          |
| 0.112         | 3.03  | 2400 | 0.1241          |
| 0.1052        | 3.16  | 2500 | 0.1234          |
| 0.103         | 3.28  | 2600 | 0.1236          |
| 0.0986        | 3.41  | 2700 | 0.1231          |
| 0.0985        | 3.54  | 2800 | 0.1232          |
| 0.0994        | 3.66  | 2900 | 0.1231          |
| 0.102         | 3.79  | 3000 | 0.1233          |
| 0.1017        | 3.91  | 3100 | 0.1228          |
| 0.0967        | 4.04  | 3200 | 0.1340          |
| 0.0841        | 4.17  | 3300 | 0.1347          |
| 0.0834        | 4.29  | 3400 | 0.1363          |
| 0.0822        | 4.42  | 3500 | 0.1346          |
| 0.0808        | 4.55  | 3600 | 0.1344          |
| 0.0845        | 4.67  | 3700 | 0.1344          |
| 0.0813        | 4.8   | 3800 | 0.1344          |
| 0.0806        | 4.92  | 3900 | 0.1345          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2