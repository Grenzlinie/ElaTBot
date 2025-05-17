from datasets import Dataset, IterableDataset
from typing import Union, Dict

def split_dataset(
    dataset: Union[Dataset, IterableDataset]
) -> Dict[str, Union[Dataset, IterableDataset]]:
    val_size = 0.05
    split_result = dataset.train_test_split(test_size=val_size, seed=42)
    return {"train_dataset": split_result["train"], "eval_dataset": split_result["test"]}

if __name__ == "__main__":
    dataset = Dataset.from_json("only_comp_desc_train_with_mpid.json")
    # Split the dataset into train and validation sets
    split_datasets = split_dataset(dataset)
    print(f"Train dataset size: {len(split_datasets['train_dataset'])}")
    print(f"Validation dataset size: {len(split_datasets['eval_dataset'])}")
    # Save the train and validation sets to disk
    split_datasets["train_dataset"].to_json("real_train_dataset_with_mpid.json")
    split_datasets["eval_dataset"].to_json("real_val_dataset_with_mpid.json")