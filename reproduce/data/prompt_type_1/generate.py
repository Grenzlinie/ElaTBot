import json
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.elasticity import ElasticTensor
import numpy as np
import math

def preserve_decimal_places(crystal_structure, decimal_places=6):
    crystal_structure = crystal_structure.as_dict()
    # 格式化晶格矩阵
    crystal_structure['lattice']['matrix'] = [[0.0 if round(value, decimal_places) == -0.0 else round(value, decimal_places) for value in row] for row in crystal_structure['lattice']['matrix']]
    crystal_structure['lattice']['a'] = 0.0 if round(crystal_structure['lattice']['a'], decimal_places) == -0.0 else round(crystal_structure['lattice']['a'], decimal_places)
    crystal_structure['lattice']['b'] = 0.0 if round(crystal_structure['lattice']['b'], decimal_places) == -0.0 else round(crystal_structure['lattice']['b'], decimal_places)
    crystal_structure['lattice']['c'] = 0.0 if round(crystal_structure['lattice']['c'], decimal_places) == -0.0 else round(crystal_structure['lattice']['c'], decimal_places)
    crystal_structure['lattice']['alpha'] = 0.0 if round(crystal_structure['lattice']['alpha'], decimal_places) == -0.0 else round(crystal_structure['lattice']['alpha'], decimal_places)
    crystal_structure['lattice']['beta'] = 0.0 if round(crystal_structure['lattice']['beta'], decimal_places) == -0.0 else round(crystal_structure['lattice']['beta'], decimal_places)
    crystal_structure['lattice']['gamma'] = 0.0 if round(crystal_structure['lattice']['gamma'], decimal_places) == -0.0 else round(crystal_structure['lattice']['gamma'], decimal_places)
    crystal_structure['lattice']['volume'] = 0.0 if round(crystal_structure['lattice']['volume'], decimal_places) == -0.0 else round(crystal_structure['lattice']['volume'], decimal_places)

    # 格式化原子坐标
    for site in crystal_structure['sites']:
        site['abc'] = [0.0 if round(coord, decimal_places) == -0.0 else round(coord, decimal_places) for coord in site['abc']]
        site['xyz'] = [0.0 if round(coord, decimal_places) == -0.0 else round(coord, decimal_places) for coord in site['xyz']]

    return crystal_structure
    

def get_pymatgen_structure(datapoint : dict) -> Structure:
    structure = Structure.from_dict(datapoint['structure'])
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
    # primitive_structure = analyzer.get_primitive_standard_structure()
    conventional_structure = analyzer.get_conventional_standard_structure()
    structure = preserve_decimal_places(structure)
    conventional_structure = preserve_decimal_places(conventional_structure)
    return structure, conventional_structure

def construct_datapoint(raw_datapoint: dict):
    # Construct a datapoint from the raw data
    dic = dict()
    primitive_structure, conventional_structure = get_pymatgen_structure(raw_datapoint)

    # Extract the basic crystal structure information
    dic['material_id']: str = raw_datapoint['material_id']
    dic['elements'] : list = raw_datapoint['elements']
    dic['formula_pretty'] : str = raw_datapoint['formula_pretty']
    dic['symmetry'] : dict = {k: v for k, v in raw_datapoint['symmetry'].items() if k not in ['version']}
    dic['primitive_structure'] = primitive_structure
    dic['conventional_structure'] = conventional_structure
    
    
    # Extract the elastic tensor
    dic['elastic_tensor'] : list or None = raw_datapoint['elastic_tensor']['ieee_format'] if raw_datapoint['elastic_tensor']['ieee_format'] else None
    # Traverse the 6x6 2D array elastic_tensor and change all -0.0 to 0.0
    if dic['elastic_tensor']:
        for i in range(6):
            for j in range(6):
                if dic['elastic_tensor'][i][j] == -0.0:
                    dic['elastic_tensor'][i][j] = 0.0
    # Extract other elasticity properties
    dic['bulk_modulus'] : dict or None = raw_datapoint['bulk_modulus'] if raw_datapoint['bulk_modulus'] else None
    dic['shear_modulus'] : dict or None = raw_datapoint['shear_modulus'] if raw_datapoint['shear_modulus'] else None
    dic['young_modulus'] : float or None = raw_datapoint['young_modulus'] if raw_datapoint['young_modulus'] else None
    dic['universal_anisotropy'] : float or None = raw_datapoint['universal_anisotropy'] if raw_datapoint['universal_anisotropy'] else None
    dic['isotropic_possion_ratio'] : float or None = raw_datapoint['homogeneous_poisson'] if raw_datapoint['homogeneous_poisson'] else None

    return dic

def construct_dataset(raw_data):
    dataset = [construct_datapoint(data) for data in raw_data]
    print(f"Dataset constructed with {len(dataset)} data points")
    return dataset

def store_dataset(dataset, stored_file_name):
    with open(stored_file_name, 'w') as f:
        json.dump(dataset, f)
    return print(f"Dataset stored in {stored_file_name}")

def calculate_properties(stiffness_matrix: list) -> (float, float, float):
    elastic_constant = np.asarray(stiffness_matrix)
    elastic_tensor = ElasticTensor.from_voigt(elastic_constant)
    youngs_modulus = round(elastic_tensor.y_mod / 1e9, 3)
    return youngs_modulus

def fix_some_keys(re_item):
    """remove unnecessary keys and rename some keys"""
    re_item = {'material_formula': re_item['formula_pretty'], **re_item}
    del re_item['material_id']
    del re_item['elements']
    del re_item['formula_pretty']
    del re_item['primitive_structure']
    del re_item['conventional_structure']['@module']
    del re_item['conventional_structure']['@class']
    del re_item['conventional_structure']['charge']
    del re_item['conventional_structure']['lattice']['pbc']
    del re_item['conventional_structure']['properties']
    for site in re_item['conventional_structure']['sites']:
        del site['abc']
        del site['properties']
        del site['label']
    del re_item['label']
    re_item = {key: value for key, value in re_item.items() if key not in ['elastic_tensor','bulk_modulus', 'shear_modulus', 'young_modulus', 'universal_anisotropy', 'isotropic_possion_ratio']}
    return re_item

# Construct the alpaca format dataset
# The alpaca format dataset is a list of dictionaries, each dictionary contains three keys: instruction, input, and output.
# The instruction means the system prompt, the input means the material information, and the output means the elastic tensor.
def build_alpaca_dataset(data):
    train_dataset = []
    test_dataset = []
    train_dataset_with_mpid = []
    test_dataset_with_mpid = []
    instruction = f"""Given a material's symmetry and conventional cell structure, predict the elastic tensor of it directly and accurately with scientific logic. Answer without any other comments, descriptions, or explanations. The answer should be a 6x6 Python matrix. The material information is presented in JSON format. """
    for item in data:
        label = item['label']
        input_item = f"Information JSON of Material {item['formula_pretty']}:" + f"{fix_some_keys(item)}"
        output_item = f"{item['elastic_tensor']}"
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
    # Load the JSON data
    with open('../mp_elastic_stable.json', 'r') as file:
        data = json.load(file)

    with open('../mp_elastic_unstable.json', 'r') as file:
        data_unstable = json.load(file)
    
    # Extract necessary information for constructing the textual input (Prompt-type 1) from Materials Project data
    stable_dataset = construct_dataset(data)
    unstable_dataset = construct_dataset(data_unstable)
    origin_data = stable_dataset + unstable_dataset

    print("Original data length", len(origin_data))
    
    # Delete the data that the moduli are unable to be calculated
    id_list = []
    for i in range(len(origin_data)):
        try:
            elastic_constant = np.asarray(origin_data[i]['elastic_tensor'])
            elastic_tensor = ElasticTensor.from_voigt(elastic_constant)
            y_m = round(elastic_tensor.y_mod / 1e9, 3)
        except Exception as e:
            id_list.append(i)

    print(id_list)
    print(len(id_list))

    for index in sorted(id_list, reverse=True):
        del origin_data[index]

    print("After removing data length", len(origin_data))
    
    # Delete the data that the bulk_modulus and shear_modulus are too large or smaller than 0.
    print("Original data length", len(origin_data))
    delete_index = []
    for i in range(len(origin_data)):
        if origin_data[i]['bulk_modulus'] == None or origin_data[i]['shear_modulus'] == None:
            delete_index.append(i)
        elif any(value < 0 for value in origin_data[i]['bulk_modulus'].values()) or any(value > 1000 for value in origin_data[i]['bulk_modulus'].values()):
            delete_index.append(i)
        elif any(value < 0 for value in origin_data[i]['shear_modulus'].values()) or any(value > 1000 for value in origin_data[i]['shear_modulus'].values()):
            delete_index.append(i)

    for index in sorted(delete_index, reverse=True):
        del origin_data[index]

    print("Deleted data with bulk_modulus or shear_modulus abnormality", len(origin_data))
    
    after_deleted_data = origin_data
    
    # Calculate the Young's modulus
    young_modulus = [calculate_properties(item['elastic_tensor']) for item in after_deleted_data]
    
    for i in range(len(after_deleted_data)):
        after_deleted_data[i]['young_modulus'] = young_modulus[i]
        
    # Split the dataset into training and testing sets
    # 5% of each crystal_system for testing
    # 95% of each crystal_system for training
    crystal_structure_data = [item['symmetry']['crystal_system'] for item in after_deleted_data]
    crystal_structure_set = set(crystal_structure_data)
    crystal_structure_count = len(crystal_structure_set)
    crystal_structure_values_count = {structure: crystal_structure_data.count(structure) for structure in crystal_structure_set}
    print(f'Crystal structures: {crystal_structure_count}')
    print(f'Crystal structure values count: {crystal_structure_values_count}')
    csv_p5 = {k: math.floor(v*0.05) for k, v in crystal_structure_values_count.items() if v >= 5}
    print(f'5% for each crystal_system: {csv_p5}')

    for k, v in csv_p5.items():
        for item in after_deleted_data:
            if item['symmetry']['crystal_system'] == k:
                item['label'] = 'test'
                v -= 1
                if v == 0:
                    break

    for item in after_deleted_data:
        if 'label' not in item:
            item['label'] = 'train'

    test_label_count = sum(1 for item in after_deleted_data if item.get('label') == 'test')
    print(f"Number of items labeled as 'test': {test_label_count}")
    
    labeled_data = after_deleted_data
    
    tr, te, trmpid, tempid = build_alpaca_dataset(labeled_data)


    with open('ec_short_train_dataset.json', 'w') as file:
        json.dump(tr, file)

    with open('ec_short_test_dataset.json', 'w') as file:
        json.dump(te, file)
        
    with open('ec_short_train_dataset_with_mpid.json', 'w') as file:
        json.dump(trmpid, file)

    with open('ec_short_test_dataset_with_mpid.json', 'w') as file:
        json.dump(tempid, file)
        
    print("Train and test datasets constructed")
        
    print(len(tr))
    print(len(te))
    
    print(tr[0])
    print(te[0])