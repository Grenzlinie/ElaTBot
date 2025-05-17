import json
from pymatgen.analysis.elasticity import ElasticTensor
import math
import random
from datasets import Dataset

def get_elastic_tensor(raw_datapoint: dict):
    return ElasticTensor.from_voigt(raw_datapoint['elastic_tensor']['ieee_format']).tolist()

def get_structure(raw_datapoint: dict):
    return raw_datapoint['structure']

def construct_datapoint(raw_datapoint: dict):
    dic = {}
    dic['crystal_system'] : str = raw_datapoint['symmetry']['crystal_system']
    dic['bulk modulus'] : dict or None = raw_datapoint['bulk_modulus'] if raw_datapoint['bulk_modulus'] else None
    dic['shear modulus'] : dict or None = raw_datapoint['shear_modulus'] if raw_datapoint['shear_modulus'] else None
    dic['young modulus'] : float or None = raw_datapoint['young_modulus'] if raw_datapoint['young_modulus'] else None
    return dic

def construct_dataset(raw_data):
    dataset = {'structure': {}, 'elastic_tensor_full': {}, 'prop_dict': {}, 'mpid': {}}
    for i in range(len(raw_data)):
        dataset['structure'][str(i)] = get_structure(raw_data[i])
        dataset['elastic_tensor_full'][str(i)] = get_elastic_tensor(raw_data[i])
        dataset['prop_dict'][str(i)] = construct_datapoint(raw_data[i])
        dataset['mpid'][str(i)] = raw_data[i]['material_id']
    print(f"Dataset constructed with {len(dataset)} data points")
    return dataset

def remove_items(dataset, id_list):
    for i in id_list:
        dataset['structure'].pop(str(i))
        dataset['elastic_tensor_full'].pop(str(i))
        dataset['prop_dict'].pop(str(i))
    return dataset

if __name__ == "__main__":
    with open('../mp_elastic_combined.json', 'r') as file:
        origin_data = json.load(file)
    training_set = Dataset.from_json("../../../reproduce/data/prompt_type_1/real_train_dataset_with_mpid.json")
    validation_set = Dataset.from_json("../../../reproduce/data/prompt_type_1/real_val_dataset_with_mpid.json")
    with open('../prompt_type_1/ec_short_test_dataset_with_mpid.json', "r") as file:
        test_set = json.load(file)
    training_set_mpids = [k['material_id'] for k in training_set]
    validation_set_mpids = [k['material_id'] for k in validation_set]
    test_set_mpids = [k['material_id'] for k in test_set]
        
    combined_dataset = construct_dataset(origin_data)
    print(len(combined_dataset['structure']))

    
    train_dataset = {'structure': {}, 'elastic_tensor_full': {}, 'prop_dict': {}, 'mpid': {}}
    test_dataset = {'structure': {}, 'elastic_tensor_full': {}, 'prop_dict': {}, 'mpid': {}}
    val_dataset = {'structure': {}, 'elastic_tensor_full': {}, 'prop_dict': {}, 'mpid': {}}

    for i in combined_dataset['prop_dict'].keys():
        if combined_dataset['mpid'][str(i)] in training_set_mpids:
            train_dataset['structure'][str(i)] = combined_dataset['structure'][str(i)]
            train_dataset['elastic_tensor_full'][str(i)] = combined_dataset['elastic_tensor_full'][str(i)]
            train_dataset['prop_dict'][str(i)] = combined_dataset['prop_dict'][str(i)]
            train_dataset['mpid'][str(i)] = combined_dataset['mpid'][str(i)]
        elif combined_dataset['mpid'][str(i)] in test_set_mpids:
            test_dataset['structure'][str(i)] = combined_dataset['structure'][str(i)]
            test_dataset['elastic_tensor_full'][str(i)] = combined_dataset['elastic_tensor_full'][str(i)]
            test_dataset['prop_dict'][str(i)] = combined_dataset['prop_dict'][str(i)]
            test_dataset['mpid'][str(i)] = combined_dataset['mpid'][str(i)]
        elif combined_dataset['mpid'][str(i)] in validation_set_mpids:
            val_dataset['structure'][str(i)] = combined_dataset['structure'][str(i)]
            val_dataset['elastic_tensor_full'][str(i)] = combined_dataset['elastic_tensor_full'][str(i)]
            val_dataset['prop_dict'][str(i)] = combined_dataset['prop_dict'][str(i)]
            val_dataset['mpid'][str(i)] = combined_dataset['mpid'][str(i)]
            
        
    print(f"Train dataset: {len(train_dataset['prop_dict'])}")
    print(f"Test dataset: {len(test_dataset['prop_dict'])}")
    print(f"Val dataset: {len(val_dataset['prop_dict'])}")
    

    
    with open('train_dataset_with_mpid.json', 'w') as f:
        json.dump(train_dataset, f, indent=3)
    with open('validation_dataset_with_mpid.json', 'w') as f:
        json.dump(val_dataset, f, indent=3)
    with open('test_dataset_with_mpid.json', 'w') as f:
        json.dump(test_dataset, f, indent=3)
    with open('train_dataset_without_mpid.json', 'w') as f:  
        train_dataset.pop('mpid')
        json.dump(train_dataset, f, indent=3)
    with open('validation_dataset_without_mpid.json', 'w') as f:
        val_dataset.pop('mpid')
        json.dump(val_dataset, f, indent=3)
    with open('test_dataset_without_mpid.json', 'w') as f:
        test_dataset.pop('mpid')
        json.dump(test_dataset, f, indent=3)
    print("Data generation completed.") 
    
    


    
    
    