---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /bohr/darwin7b-mjsq/v1/darwin-7b
model-index:
- name: darwin_desc_short
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# darwin_desc_short

This model is a fine-tuned version of [/bohr/darwin7b-mjsq/v1/darwin-7b](https://huggingface.co//bohr/darwin7b-mjsq/v1/darwin-7b) on the ec_short_train_dataset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1782

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
| 0.3041        | 0.13  | 100  | 0.2467          |
| 0.2156        | 0.25  | 200  | 0.2083          |
| 0.2039        | 0.38  | 300  | 0.1981          |
| 0.1921        | 0.51  | 400  | 0.1938          |
| 0.1887        | 0.63  | 500  | 0.1886          |
| 0.1866        | 0.76  | 600  | 0.1869          |
| 0.1866        | 0.88  | 700  | 0.1844          |
| 0.1817        | 1.01  | 800  | 0.1828          |
| 0.1814        | 1.14  | 900  | 0.1824          |
| 0.1781        | 1.26  | 1000 | 0.1799          |
| 0.1811        | 1.39  | 1100 | 0.1785          |
| 0.1763        | 1.52  | 1200 | 0.1775          |
| 0.1682        | 1.64  | 1300 | 0.1754          |
| 0.1722        | 1.77  | 1400 | 0.1751          |
| 0.1668        | 1.89  | 1500 | 0.1752          |
| 0.1704        | 2.02  | 1600 | 0.1733          |
| 0.1678        | 2.15  | 1700 | 0.1726          |
| 0.1662        | 2.27  | 1800 | 0.1730          |
| 0.163         | 2.4   | 1900 | 0.1724          |
| 0.1654        | 2.53  | 2000 | 0.1706          |
| 0.1642        | 2.65  | 2100 | 0.1706          |
| 0.1608        | 2.78  | 2200 | 0.1705          |
| 0.1633        | 2.9   | 2300 | 0.1695          |
| 0.1645        | 3.03  | 2400 | 0.1702          |
| 0.1514        | 3.16  | 2500 | 0.1715          |
| 0.1555        | 3.28  | 2600 | 0.1713          |
| 0.1475        | 3.41  | 2700 | 0.1704          |
| 0.147         | 3.54  | 2800 | 0.1702          |
| 0.1516        | 3.66  | 2900 | 0.1700          |
| 0.1529        | 3.79  | 3000 | 0.1701          |
| 0.1495        | 3.91  | 3100 | 0.1698          |
| 0.1435        | 4.04  | 3200 | 0.1765          |
| 0.1366        | 4.17  | 3300 | 0.1771          |
| 0.1355        | 4.29  | 3400 | 0.1788          |
| 0.1321        | 4.42  | 3500 | 0.1791          |
| 0.1287        | 4.55  | 3600 | 0.1775          |
| 0.1353        | 4.67  | 3700 | 0.1777          |
| 0.1326        | 4.8   | 3800 | 0.1780          |
| 0.1312        | 4.92  | 3900 | 0.1782          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2