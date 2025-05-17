import json
from pymatgen.analysis.elasticity import ElasticTensor
import numpy as np
import math

def construct_datapoint(raw_datapoint: dict):
    dic = {}
    dic['material_id']: str = raw_datapoint['material_id']
    dic['crystal system'] : str = raw_datapoint['symmetry']['crystal_system']
    dic['description'] : str = raw_datapoint['description']
    # Extract the elastic tensor
    dic['elastic tensor'] : list or None = raw_datapoint['elastic_tensor']['ieee_format'] if raw_datapoint['elastic_tensor']['ieee_format'] else None
    # Traverse the 6x6 2D array elastic_tensor and change all -0.0 to 0.0
    if dic['elastic tensor']:
        for i in range(6):
            for j in range(6):
                if dic['elastic tensor'][i][j] == -0.0:
                    dic['elastic tensor'][i][j] = 0.0
    # Extract other elasticity properties
    dic['bulk modulus'] : dict or None = raw_datapoint['bulk_modulus'] if raw_datapoint['bulk_modulus'] else None
    dic['shear modulus'] : dict or None = raw_datapoint['shear_modulus'] if raw_datapoint['shear_modulus'] else None
    dic['young modulus'] : float or None = raw_datapoint['young_modulus'] if raw_datapoint['young_modulus'] else None
    dic['universal anisotropy'] : float or None = raw_datapoint['universal_anisotropy'] if raw_datapoint['universal_anisotropy'] else None
    dic['isotropic possion ratio'] : float or None = raw_datapoint['homogeneous_poisson'] if raw_datapoint['homogeneous_poisson'] else None

    return dic

def construct_dataset(raw_data):
    dataset = [construct_datapoint(data) for data in raw_data]
    print(f"Dataset constructed with {len(dataset)} data points")
    return dataset

def calculate_properties(stiffness_matrix: list) -> (float, float, float):
    elastic_constant = np.asarray(stiffness_matrix)
    elastic_tensor = ElasticTensor.from_voigt(elastic_constant)
    youngs_modulus = round(elastic_tensor.y_mod / 1e9, 3)
    return youngs_modulus

# Construct the alpaca dataset
def build_alpaca_dataset(data):
    train_dataset = []
    test_dataset = []
    train_dataset_with_mpid = []
    test_dataset_with_mpid = []
    instruction = "Given a material's crystal structure description, predict the elastic tensor of it directly and accurately with scientific logic. Answer without any other comments, descriptions, or explanations. The answer should be a 6x6 Python matrix. "
    for item in data:
        label = item['label']
        input_item = item['description']
        output_item = f"{item['elastic tensor']}"
        if label == 'test':
            test_dataset.append({
                'instruction': instruction,
                'input': input_item,
                'output': output_item,
            })
            test_dataset_with_mpid.append({
                'instruction': instruction,
                'input': input_item,
                'output': output_item,
                'material_id': item['material_id'],
            })
        else:
            train_dataset.append({
                'instruction': instruction,
                'input': input_item,
                'output': output_item,
            })
            train_dataset_with_mpid.append({
                'instruction': instruction,
                'input': input_item,
                'output': output_item,
                'material_id': item['material_id'],
            })
    return train_dataset, test_dataset, train_dataset_with_mpid, test_dataset_with_mpid

if __name__ == "__main__":
    with open("../mp_elastic_combined.json", "r") as f:
        data = json.load(f)
    combined_dataset = construct_dataset(data)
    origin_data = combined_dataset
    
    print("Original data length", len(origin_data))
    id_list = []
    for i in range(len(origin_data)):
        try:
            elastic_constant = np.asarray(origin_data[i]['elastic tensor'])
            elastic_tensor = ElasticTensor.from_voigt(elastic_constant)
            y_m = round(elastic_tensor.y_mod / 1e9, 3)
        except Exception as e:
            id_list.append(i)

    print(id_list)
    print(len(id_list))
    
    for index in sorted(id_list, reverse=True):
        del origin_data[index]

    print("After removing data length", len(origin_data))
    
    # Delete data with abnormal bulk modulus and shear modulus
    print("Original data length", len(origin_data))
    delete_index = []
    for i in range(len(origin_data)):
        if origin_data[i]['bulk modulus'] == None or origin_data[i]['shear modulus'] == None:
            delete_index.append(i)
        elif any(value < 0 for value in origin_data[i]['bulk modulus'].values()) or any(value > 1000 for value in origin_data[i]['bulk modulus'].values()):
            delete_index.append(i)
        elif any(value < 0 for value in origin_data[i]['shear modulus'].values()) or any(value > 1000 for value in origin_data[i]['shear modulus'].values()):
            delete_index.append(i)

    for index in sorted(delete_index, reverse=True):
        del origin_data[index]

    print("Deleted data with bulk_modulus or shear_modulus abnormality", len(origin_data))
    
    after_deleted_data = origin_data
    bulk_modulus = [item['bulk modulus']['vrh'] for item in after_deleted_data]
    shear_modulus = [item['shear modulus']['vrh'] for item in after_deleted_data]
    young_modulus = [calculate_properties(item['elastic tensor']) for item in after_deleted_data]

    for i in range(len(after_deleted_data)):
        after_deleted_data[i]['young modulus'] = young_modulus[i]
    
    data = after_deleted_data
        
    crystal_structure_data = [item['crystal system'] for item in data]
    crystal_structure_set = set(crystal_structure_data)
    crystal_structure_count = len(crystal_structure_set)
    crystal_structure_values_count = {structure: crystal_structure_data.count(structure) for structure in crystal_structure_set}
    print(f'Crystal structures: {crystal_structure_count}')
    print(f'Crystal structure values count: {crystal_structure_values_count}')
    csv_p5 = {k: math.floor(v*0.05) for k, v in crystal_structure_values_count.items() if v >= 5}
    print(f'5% for each crystal_system: {csv_p5}')

    for k, v in csv_p5.items():
        for item in data:
            if item['crystal system'] == k:
                item['label'] = 'test'
                v -= 1
                if v == 0:
                    break

    for item in data:
        if 'label' not in item:
            item['label'] = 'train'
    
    for k, v in data[0].items():
        print(k, v)
        
    tr, te, trmpid, tempid = build_alpaca_dataset(data)

    with open('only_structure_desc_train.json', 'w') as file:
        json.dump(tr, file)

    with open('only_structure_desc_test.json', 'w') as file:
        json.dump(te, file)
    
    with open('only_structure_desc_train_with_mpid.json', 'w') as file:
        json.dump(trmpid, file)

    with open('only_structure_desc_test_with_mpid.json', 'w') as file:
        json.dump(tempid, file)
        
    print("Train and test datasets constructed")
        
    print(len(tr), len(te))
    print(tr[0])
    print(te[0])