---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/
model-index:
- name: prompt1_special
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# prompt1_special

This model is a fine-tuned version of [/bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/](https://huggingface.co//bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/) on the ec_short_train_dataset_special dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1381

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
| 0.1602        | 0.13  | 100  | 0.1460          |
| 0.142         | 0.25  | 200  | 0.1392          |
| 0.1389        | 0.38  | 300  | 0.1349          |
| 0.1327        | 0.51  | 400  | 0.1347          |
| 0.133         | 0.63  | 500  | 0.1329          |
| 0.1338        | 0.76  | 600  | 0.1322          |
| 0.1317        | 0.88  | 700  | 0.1298          |
| 0.1285        | 1.01  | 800  | 0.1303          |
| 0.1253        | 1.14  | 900  | 0.1290          |
| 0.1273        | 1.26  | 1000 | 0.1273          |
| 0.1269        | 1.39  | 1100 | 0.1275          |
| 0.1259        | 1.52  | 1200 | 0.1266          |
| 0.1231        | 1.64  | 1300 | 0.1252          |
| 0.1222        | 1.77  | 1400 | 0.1240          |
| 0.1198        | 1.89  | 1500 | 0.1240          |
| 0.1226        | 2.02  | 1600 | 0.1232          |
| 0.1184        | 2.15  | 1700 | 0.1241          |
| 0.1163        | 2.27  | 1800 | 0.1239          |
| 0.1134        | 2.4   | 1900 | 0.1223          |
| 0.1133        | 2.53  | 2000 | 0.1220          |
| 0.114         | 2.65  | 2100 | 0.1220          |
| 0.1122        | 2.78  | 2200 | 0.1206          |
| 0.1117        | 2.9   | 2300 | 0.1203          |
| 0.1108        | 3.03  | 2400 | 0.1254          |
| 0.102         | 3.16  | 2500 | 0.1258          |
| 0.1002        | 3.28  | 2600 | 0.1253          |
| 0.0961        | 3.41  | 2700 | 0.1244          |
| 0.0963        | 3.54  | 2800 | 0.1254          |
| 0.0974        | 3.66  | 2900 | 0.1249          |
| 0.0997        | 3.79  | 3000 | 0.1253          |
| 0.0986        | 3.91  | 3100 | 0.1243          |
| 0.0941        | 4.04  | 3200 | 0.1381          |
| 0.0796        | 4.17  | 3300 | 0.1384          |
| 0.0782        | 4.29  | 3400 | 0.1389          |
| 0.0775        | 4.42  | 3500 | 0.1383          |
| 0.0756        | 4.55  | 3600 | 0.1379          |
| 0.0792        | 4.67  | 3700 | 0.1379          |
| 0.0759        | 4.8   | 3800 | 0.1380          |
| 0.0748        | 4.92  | 3900 | 0.1381          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2