# Large Language Models for Material Property Predictions: elastic constant tensor prediction and materials design
Here is the source code related to the [article](https://arxiv.org/abs/2411.12280).  

For the complete code and data files, please refer to: [https://doi.org/10.6084/m9.figshare.28399757.v1](https://doi.org/10.6084/m9.figshare.28399757.v1)

# Reproduction Steps
We provide a full repoduction guide with `reproduce` folder. Here are these steps.
## 0. Environment Configuration
> Before we start downloading data, processing data, training model, building agent and analysis results, it's necessary to configure the python environment.

```plain text
Requirements: A computer equipped with NVIDIA GPU, with at least 24GB of graphics memory. It is recommended to use a Linux operating system for training.
```

### 0.1 Create environment for training part
We test these setting and scripts on our Macbook with M1 Chip and on Debian system with CUDA 12.1 and 4 x NVIDIA A100 40GB.

```bash
# create the conda environment
conda create -n llm python=3.10.14

# activate environment
conda activate llm

# install packages
pip install -r requirements.txt

# check the torch availability
python

>>> import torch
# for mac
>>> torch.backends.mps.is_available()
# for linux or windows
>>> torch.cuda.is_available()
# >>> True
>>> exit()

conda deactivate
```

### 0.2 Create environment for RAG and Agent UI:
```bash
# create the conda environment
conda create -n robot python=3.11.9
# activate environment
conda activate robot
# install packages
pip install -r requirements_robot.txt
pip install -U langchain_openai==0.1.3
pip install -U bitsandbytes
conda deactivate
```

## 1. Prepare input and output data for training
We test these setting and scripts on our Macbook with M1 Chip.

```bash
conda activate llm # for step 1.
```
These steps help us prepare the training data of ElaTBot-DFT.
### 1.1 (Optional) Download materials data with elastic constant tensor from Materials Project

If it is only for reproduction, please skip this step.

(Note: The official Materials Project (MP) API currently does not support downloading data for a specified database version. e.g., our version: 2023.11.01, link: https://materialsproject-build.s3.amazonaws.com/index.html#collections/2023-11-01/. As a result, running this code will download data from the latest available version (2025). To ensure reproducibility, please use the 2023 version dataset that we provided and skip this step.)

(Note: The latest mp-api version doesn't compatible with our pymatgen version, so we didn't provide mp-api version in `requirements.txt` for downloading data. If you want to download data, please install mp-api and then uninstall it and then reinstall pymatgen==2023.12.18 for later steps.)

open `reproduce/data/download_mp.ipynb`, run the first block. Each entry in `mp_elastic_stable.json` and `mp_elastic_unstable.json` is a material with unique material_id.

### 1.2 Generate crystal structure description with robocrystallographer
run the remain blocks at `reproduce/data/download_mp.ipynb`. (Note: the generation process will take a while. We used `mp_elastic_stable_with_desc.json` and `mp_elastic_unstable_with_desc.json` to get `mp_elastic_combined.json` (This file just adds the crystal structure description by using robocrystallographer while the same file in `Data/dft_dataset/download_data` doesn't contain, this change is to help us clean the code. We checked these two file are same by using `Data/dft_dataset/download_data/check_data.ipynb`))
 
Now we will have five json files in the folder.
| File name  | Introduction |
|----------------|--------------------------------------------------------------|
| mp_elastic_stable.json     | downloaded stable materials data from materials project |
| mp_elastic_unstable.json      | downloaded unstable materials data from materials project |
| mp_elastic_stable_with_desc.json      | downloaded stable materials data with textual crystal structure description |
| mp_elastic_unstable_with_desc.json      | downloaded unstable materials data with textual crystal structure description |
| mp_elastic_combined.json | data that combines mp_elastic_stable_with_desc.json and mp_elastic_unstable_with_desc.json |

### 1.3 Generate input and output data for different methods
> Compared with original data, we added material_id to make sure the consistency for each type of data when they are splitted to train/val/test. And we checked the consistency of data in `reproduce/data` and `Data` folders by using `Data/dft_dataset/download_data/check_data.ipynb`. All data is consistent, even if they are divided into different subsets. The user can also rerun this notebook to see the checking result.

- ElaTBot-DFT
1. Prompt type 1
```bash
cd reproduce/data/prompt_type_1
python generate.py # get the splitted training/test dataset, the validation set will be splitted by Llama-Factory from training set with fixed random seed 42.
# It is normal for pymatgen Warning to display during operation, which doesn't affect our code execution.
```

Regarding the division of validation set:
In early versions of LLaMA-Factory, the validation set partitioning during the training phase was specified through data_args.val_size, rather than manually passing in the validation set. Therefore, the training set generated through generate.py is actually a combination of training and validation sets. During actual training, LLaMA-Factory partitions the input training set using the following function.

According to our settings, 95% of the input training set is the actual training set and 5% is the validation set.
```python
# the splitting method of training and validation set:
# ElaTBoT-DFT/LLaMA-Factory/src/llmtuner/data/utils.py
# training_args.seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."}) so we use the default seed 42.
# And the validation size we set: data_args.val_size 0.05 in `ElaTBoT-DFT/train_bash/desc_llama.sh`.
def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", training_args: "Seq2SeqTrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6:
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else: # !According to our parameters, we will enter this situation
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset": dataset}
    else:  # do_eval or do_predict
        return {"eval_dataset": dataset}
```

We simulated the process of the division of training and validation set:
```bash
python simulate_split_validation.py
```
You will get two files: `real_train_dataset_with_mpid.json` and `real_val_dataset_with_mpid.json`. This can help you check if different prompt types have the same train/val/test division by comparing material_id.

2. Prompt type 2
```bash
cd reproduce/data/prompt_type_2
python generate.py # get the splitted training/test dataset, the validation set will be splitted by Llama-Factory from training set with fixed random seed.
```
Also,
```bash
python simulate_split_validation.py
```

3. Prompt type 3
```bash
cd reproduce/data/prompt_type_3
python generate.py # get the splitted training/test dataset, the validation set will be splitted by Llama-Factory from training set with fixed random seed.
```
Also,
```bash
python simulate_split_validation.py
```

4. Prompt type 4
```bash
cd reproduce/data/prompt_type_4
python generate.py # get the splitted training/test dataset, the validation set will be splitted by Llama-Factory from training set with fixed random seed.
```
Also,
```bash
python simulate_split_validation.py
```

5. MatTen
```bash
cd reproduce/data/matten
python generate.py # get the splitted training/validation/test dataset with the same mp_ids of prompt type 1/2/3/4. 
```

6. Random Forest
```bash
cd reproduce/data/random_forest # get the splitted training/validation/test dataset with the same mp_ids of prompt type 1/2/3/4.
```
run all blocks in `random_forest.ipynb`

7. Darwin
Use the same dataset with prompt type 4.

- ElaTBoT

```bash
cd reproduce/data/mixed_dataset
python generate.py # Generate preprocessing content of DFT dataset
```
run `deal_mp_desc_data.ipynb` to generate datapoints for tasks to predict 0K elastic tensor and bulk modulus

run `deal_reverse_data.ipynb` to generate datapoints for infilling task and materials generation task

run `deal_temp_data.ipynb` to generate datapoints for predicting finite temperature elastic tensor and bulk modulus

- (Optional) Check data consistency
open `Data/dft_dataset/download_data/check_data.ipynb` and run all blocks.

### 1.4 Data description
| File name  | Introduction |
|----------------|--------------------------------------------------------------|
| reproduce/data/prompt_type_1/ec_short_train_dataset.json     | Input training set for prompt_type_1 (validation set will be splitted by LLaMA-Factory) |
| reproduce/data/prompt_type_1/ec_short_test_dataset.json      | Test set for prompt_type_1     |
| reproduce/data/prompt_type_2/only_structure_desc_train.json  | Input training set for prompt_type_2 (validation set will be splitted by LLaMA-Factory)   |
| reproduce/data/prompt_type_2/only_structure_desc_test.json  | Test set for prompt_type_2   |
| reproduce/data/prompt_type_3/only_comp_desc_train.json  | Input training set for prompt_type_3 (validation set will be splitted by LLaMA-Factory)   |
| reproduce/data/prompt_type_3/only_comp_desc_test.json  | Test set for prompt_type_3   |
| reproduce/data/prompt_type_4/ec_desc_train_dataset.json  | Input training set for prompt_type_4 and Darwin (validation set will be splitted by LLaMA-Factory)   |
| reproduce/data/prompt_type_4/ec_desc_test_dataset.json  | Test set for prompt_type_4 and Darwin  |
| reproduce/data/matten/train_dataset_without_mpid.json  | Training set for MatTen   |
| reproduce/data/matten/validation_dataset_without_mpid.json  | Validation set for MatTen |
| reproduce/data/matten/test_dataset_without_mpid.json  | Test set for MatTen   |
| reproduce/data/random_forest/training_data.json | Training set for random forest model. | 
| reproduce/data/random_forest/test_data.json | Test set for random forest model. | 
| reproduce/data/random_forest/validation_data.json | Validation set for random forest model. |
| reproduce/data/mixed_dataset/combined_data.json | Training set for ElaTBot. |
| reproduce/data/temp_data/combined_temp_data.json | Original finite temperature data. We ignore temp_data[i]['pressure'] != 1 or temp_data[i]['temperature'] == 0 data to ensure consistency. After the treatment, it will remain 1266 datapoints.|

Other files are intermediate files.

## 2. Training and testing ElaTBot-DFT, Darwin, MatTen, Random Forest model and ElaTBot
We test these scripts on Debian system with CUDA 12.1 and 4 x NVIDIA A100 40GB.

```bash
conda activate llm # with other description.
```
### 2.0 Download models

- Llama2-7b-chat-hf
```bash
cd Models
python download_from_huggingface.py --model_name meta-llama/Llama-2-7b-chat-hf --save_path . --token Your huggingface_api_key
```

- Darwin

Manually download to the Model folder from the following link: https://aigreendynamics-my.sharepoint.com/personal/yuwei_greendynamics_com_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuwei%5Fgreendynamics%5Fcom%5Fau%2FDocuments%2Fdarwin%2D7b&ga=1 (Note: check the parameter "_name_or_path" is equal to "/srv/scratch/z5293104/darwin/stanford_alpaca/training_output/sciq_base_mix13_training3/checkpoint-3000" in config.json)

- MatTen: 
We prepared it in `Models/matten-main`.

- Random Forest: 
We used scikit-learn to implement it. The code can be found in `reproduce/data/random_forest/random_forest.ipynb`.

### 2.1 Four prompt types and Darwin

For training (we use prompt type 1 as an example)
```bash
cd ElaTBoT-DFT/train_bash_demo
bash desc_llama.sh
sudo env PATH=$PATH:/opt/conda/envs/llm/bin bash desc_llama.sh # if face error with accelerate.
```

If you want to try other prompt type or darwin, modify the parameter of `desc_llama.sh` such as `model_name_or_path`, `dataset`, `output_dir`.

If your device is other types (e.g. different numbers of GPU), you also need modify the `num_processes` in `../LLaMA-Factory/examples/accelerate/single_config.yaml` and `CUDA_VISIBLE_DEVICES` in `desc_llama.sh`.

For test (we use prompt type 1 as an example)
```bash
cd ElaTBoT-DFT/ft_result
bash cpa.sh ./prompt1 ../adapter # move the lora weight to adapter folder
cd ../train_bash_demo
bash desc_llama_test.sh # it will automatically merge lora weight with llama model for testing.
sudo env PATH=$PATH:/opt/conda/envs/llm/bin bash desc_llama_test.sh # if face error with accelerate.
```

You will get things like results in `ElaTBoT-DFT/ft_result/prompt1`:
1. the trained lora weight files in `ElaTBoT-DFT/ft_result/prompt1`. You can run the code in step 2.5 to merge lora weight with llama model to a fine-tuned model.
2. the test result files in `ElaTBoT-DFT/ft_result/prompt1/test_result`. The predicted elastic tensors of materials are in `ElaTBoT-DFT/ft_result/prompt1/test_result/generated_predictions.jsonl`. We used this for analysis. (A condition is, after generating 522 test sets, the script of LLaMA-Facotry may start generating the first few elastic constants of the test set from scratch due to its batch processing mechanism. We only took the first 522 in jsonl file.)

3. An interesting thing is, when we run the same test script on different computers, the predicted elastic tensor in generated_predictions.jsonl may slightly differ, while remaining consistent on the same computer (we ran tests twice on each computer to check for consistency). For example, our original result (R<sup>2</sup>: 0.9473497 for predicting prompt type 1 elastic tensor) was generated on the [Bohrium platform](https://bohrium.dp.tech/home) with CUDA version 12.1 and 4 x V100 32GB. However, we obtained different results (R<sup>2</sup>: 0.9479842 for predicting prompt type 1 elastic tensor) on [Google Cloud Compute Engine](https://cloud.google.com/?hl=en) with CUDA version 12.4 and 4 x A100 40GB, and yet another result (R<sup>2</sup>: xxx for predicting prompt type 1 elastic tensor) with CUDA version 12.1 and 4 x L20 48GB. We have stored the test result folders in `ElaTBoT-DFT/ft_result/prompt1/`. We believe that due to the inherent randomness of LLM generation, even when fixing the random seed (42) and reducing the LLM temperature to 0.01 (not 0 due to limitations of LLaMA-Factory), there is still some probability of randomness due to differences in computing environments. Since the same result can be reproduced in a fixed environment, we will uniformly use the results previously obtained on the [Bohrium platform](https://bohrium.dp.tech/home) with CUDA version 12.1 and 4 x V100 32GB for our analysis.

### 2.2 MatTen
We have prepared training/test/validation sets in `Models/matten-main/datasets` by copying the datasets from `reproduce/data/matten`.

Here we use computer with CUDA version 11.8 for training to avoid dependencies error.

You just need to run these script for training.
```bash
cd Models/
conda create -n matten python=3.9
conda activate matten
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e ./matten-main
pip install -U scikit-learn
pip install numpy==1.26.4
cd ./matten-main/scripts
nohup python train_materials_tensor.py >> matten.log 2>&1 &
```

Note: At the end of the log, we see that there was an issue running the test set because matte does not support the Ne element. Therefore, for the test, we called `Models/matten-main/src/matten/predict.py` again to get the test result and skipped the Ne element, which results in a total of 521 test sets.

For testing:

1. Move the training checkpoint `epoch=199-step=158400.ckpt` in `Models/matten-main/scripts/lightning_logs/checkpoints` to `Models/matten-main/pretrained/20250412`. Change the folder name to a date like `20250412`. Change the model checkpoint name to `model_final.ckpt`. Then move `Models/matten-main/scripts/configs/config_with_20230627.yaml` to `Models/matten-main/pretrained/20250412` and change the name to `config_final.yaml`.
2. modify the `model_identifier` of predict function in `Models/matten-main/src/matten/predict.py` to the model name you set at step 1.
3. 
```bash
cd Models/matten-main/src/matten
python predict.py
```

You will get `elastic_tensors_predicted_test_set.npy` (store the elastic tensor prediction of test set), `elastic_tensors_real_test_set.npy` (store the elastic tensor of test set). We used this for analysis. We move the result files to `ElaTBoT-DFT/algorithm_comparison/MatTen_result`.


### 2.3 Random Forest 
open `ElaTBoT-DFT/algorithm_comparison/random_forest/random_forest.ipynb`

Then run all blocks for training and test. The results like prediction elastic tensor, bulk modulus and mean absolute error will show in the block output. We used the result for analysis.

### 2.4 ElaTBot training and test
```bash
conda activate llm
```

```bash
cd ElaTBoT/ft/
```

For training
```bash
cd cb
bash cb.sh
sudo env PATH=$PATH:/opt/conda/envs/llm/bin bash cb.sh # if face error with accelerate.
```

For batch generation and finite temperature prediction, we provide some input prompt templates in ElaTBoT/ft/training_tool/data, such as generating alloy compositions with different moduli and predicting elastic tensor for the three different alloys mentioned in the manuscript. You just need to replace the `--dataset`, `--output_dir` and `--temperature`(0.01 for prediction and 0.95 for generation) parameters in `ElaTBoT/ft/cb/cb_test_multigpu.sh` to get the results.

```bash
cd cb
bash cb_test_multigpu.sh
sudo env PATH=$PATH:/opt/conda/envs/llm/bin bash cb_test_multigpu.sh # if face error with accelerate.
```


### 2.5 Merge lora weight
For building the conversation agent, we need merge the lora weight after training ElaTBot.

```bash
cd lora_merge_utils
conda activate robot
pip install -e ".[torch,metrics]"
```

We will use the script `lora_merge_utils/examples/merge_lora/merge_lora.yaml` to merge lora weight. Generally speaking, you only need to replace the path before `source_code` with the absolute path on your computer.
```yaml
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters
### model
model_name_or_path: /home/wy1121535783/source_code/Models/Llama-2-7b-chat-hf # replace with your llama model path
adapter_name_or_path: /home/wy1121535783/source_code/ElaTBoT/ft_result/combined_training/ # replace with your ElaTBot LoRA weight path
template: llama2
finetuning_type: lora

### export
export_dir: /home/wy1121535783/source_code/Models/ElaTBoT # replace with your export dir path
export_size: 2
export_device: gpu
export_legacy_format: false
```

Then just run this command:
```bash
llamafactory-cli export examples/merge_lora/merge_lora.yaml # in `lora_merge_utils` folder
```

Then you will get ElaTBoT model at `source_code/Models/ElaTBoT`.

## 3. Building conversation agent
## 3.1 Merge lora weight
Run the script in step 2.5. Just make sure to use the absolute path on your computer.

## 3.2 build conversation agent

```bash
cd Conversation-Agent
conda activate robot
python llama_hand_ver.py
```

If you have openai api-key:
```bash
cd Conversation-Agent
conda activate robot
export OPENAI_API_KEY="your api key here"
# Cancel the comment on line 91 in llama_hand_ver.py ⬇️.
# vectordb = Chroma.from_documents(documents=data, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
# Generally speaking, using OpenAI for embedding results will be better.
python llama_hand_ver.py
```

You will get the URL and can open them to interact with Agent UI.

> Running on public URL: https://xxx.xxx

> Note: The performance of RAG is influenced by various factors including the embedding temperature of ElaTBot(we set 0.01), embedding model version, conversation history order, and computing environment. Our tests conducted on different servers (Bohrium, Google Cloud) did show minor variations in results. However, for a given environment, when model configurations are consistent, the responses generated during single-turn dialogue should remain consistent as well. This phenomenon also occurred during the training of Step 2.1. Because RAG is only an experimental study in our work, we don't discuss it in detail here. We will strive to study these differences clearly in our future work. Currently, we are based on the conversation screenshots stored in `Conversation-Agent/rag_decord`.

# Brief description of folders
## ElaTBot-DFT folder
This folder includes the training code, bash scripts, results of ElaTBot-DFT and the comparison models.

| File/Folder name  | Introduction |
|----------------|--------------------------------------------------------------|
| adapter     | The folder used to store LoRA weights for testing. When testing, please include the contents of different prompt folders in ft_desult. |
| algorithm_comparison     | The folder for storing the results of comparison models. |
| ft_result     | The folder for storing the results of four types prompt. |
| ElaTBoT/ft_result/cpa.sh | bash to move LoRA weight to adapter folder. |
| LLaMA-Factory     | Training tool. |
| LLaMA-Factory/data  | Training and test datasets for different prompts. |
| train_bash_demo    | Training and test bash. |

## ElaTBot folder
This folder includes the training code, results of ElaTBot and finite temperature prediction and materials generation results by using ElaTBot.

| File/Folder name  | Introduction |
|----------------|--------------------------------------------------------------|
| ft/adapter     | The folder used to store LoRA weights of ElaTBot for testing. |
| ft/training_tool     | Training tool. |
| ft/cb    | Training and test bash. |
| ft/training_tool/data | Training and test datasets. |
| ft_result/combined_training  | The folder for storing the training results and test results of ElaTBot. |

## Conversation Agent Folder
This folder contains the RAG results and agent code. It should be noted that Langchain's version updates are relatively fast. We used a very early version when completing this project, and we will try to update the new version of the Agent build in the future.

| File/Folder name  | Introduction |
|----------------|--------------------------------------------------------------|
| rag_record | Results of predicting bulk modulus with RAG |
| rag_record_all_data | Results of predicting bulk modulus with RAG (use all data to construct knowledge base) |
| without_rag_result | Results of predicting bulk modulus without RAG |
| hf_chat.py | Intermediate file used for assembling dialogue models.      |
| llama_hand_ver.py | The code file for running the Agent UI.          |
| merged_data.csv | Original database of the bulk modulus of the three alloys. |
| merged_data_for_test.csv | Database for RAG retrieval that removes 18 (9 test set and 9 random) alloys bulk modulus data.  |
| merged_data_all.csv | Database for RAG retrieval that removes 9 alloys bulk modulus data (type 1).  |


## Data Folder
- Deprecated and will be removed in the future.

## lora_merge_utils Folder
- Code for merging LoRA weight.

## reproduce Folder
- Folder for reproducing the input data and adding material_id for tracking.

## analysis Folder
- Folder for code to analyze results.

| File name  | Introduction |
|----------------|--------------------------------------------------------------|
| calc_prompt_test_result.ipynb | Calculation scripts to obtain prompt test results. |
| different.ipynb | Caculation scripts to obtain the prediction error versus crystal systems. |
| element.ipynb | Analysis scripts of data distribution by elements. |
| fig2_pictures.ipynb | Picture drawing script for Figure 2 of manuscript. |
| matten_result_analysis.ipynb | Calculation scripts to obtain MatTen test results.  |
| symmetry_analysis_test.ipynb | Analysis scripts for data distribution by crystal systems and for symmetry and stability criteria versus different models. |
| mp_data.jsonl | Intermediate data before filtering and splitting to training/val/test sets. |