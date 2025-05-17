---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/
model-index:
- name: only_comp
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# only_comp

This model is a fine-tuned version of [/bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/](https://huggingface.co//bohr/llama7bhf-elcd/v1/llama2/Llama-2-7b-chat-hf/) on the only_comp_desc_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1443

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
| 0.1601        | 0.13  | 100  | 0.1504          |
| 0.1476        | 0.25  | 200  | 0.1460          |
| 0.1454        | 0.38  | 300  | 0.1439          |
| 0.1449        | 0.51  | 400  | 0.1425          |
| 0.1409        | 0.63  | 500  | 0.1402          |
| 0.1411        | 0.76  | 600  | 0.1396          |
| 0.1386        | 0.88  | 700  | 0.1390          |
| 0.1342        | 1.01  | 800  | 0.1380          |
| 0.1322        | 1.14  | 900  | 0.1367          |
| 0.1353        | 1.26  | 1000 | 0.1366          |
| 0.1348        | 1.39  | 1100 | 0.1353          |
| 0.1342        | 1.52  | 1200 | 0.1346          |
| 0.131         | 1.64  | 1300 | 0.1337          |
| 0.1301        | 1.77  | 1400 | 0.1330          |
| 0.1278        | 1.89  | 1500 | 0.1322          |
| 0.1314        | 2.02  | 1600 | 0.1316          |
| 0.1266        | 2.15  | 1700 | 0.1313          |
| 0.1253        | 2.27  | 1800 | 0.1313          |
| 0.1218        | 2.4   | 1900 | 0.1314          |
| 0.1225        | 2.53  | 2000 | 0.1305          |
| 0.1235        | 2.65  | 2100 | 0.1295          |
| 0.1208        | 2.78  | 2200 | 0.1292          |
| 0.1206        | 2.9   | 2300 | 0.1297          |
| 0.1206        | 3.03  | 2400 | 0.1330          |
| 0.1114        | 3.16  | 2500 | 0.1346          |
| 0.1101        | 3.28  | 2600 | 0.1320          |
| 0.1059        | 3.41  | 2700 | 0.1315          |
| 0.1059        | 3.54  | 2800 | 0.1322          |
| 0.1069        | 3.66  | 2900 | 0.1318          |
| 0.1098        | 3.79  | 3000 | 0.1325          |
| 0.1093        | 3.91  | 3100 | 0.1318          |
| 0.1045        | 4.04  | 3200 | 0.1421          |
| 0.0909        | 4.17  | 3300 | 0.1429          |
| 0.0897        | 4.29  | 3400 | 0.1435          |
| 0.0885        | 4.42  | 3500 | 0.1442          |
| 0.0859        | 4.55  | 3600 | 0.1441          |
| 0.0904        | 4.67  | 3700 | 0.1442          |
| 0.0872        | 4.8   | 3800 | 0.1442          |
| 0.0868        | 4.92  | 3900 | 0.1443          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2