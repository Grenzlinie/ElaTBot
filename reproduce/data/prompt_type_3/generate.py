import json
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.elasticity import ElasticTensor
import numpy as np
import math

def append_property_info(element, property_value, property_name, unit=""):
    if property_value[element]:
        return f"{property_name} of {property_value[element]}{unit}, "
    return ""

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

#构建alpaca数据集
def build_alpaca_dataset(data):
    train_dataset = []
    test_dataset = []
    train_dataset_with_mpid = []
    test_dataset_with_mpid = []
    instruction = "Given a material's composition description, predict the elastic tensor of it directly and accurately with scientific logic. Answer without any other comments, descriptions, or explanations. The answer should be a 6x6 Python matrix. "
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
        
    # 获取所有元素的电负性 Get electronegativity of all elements
    electronegativities = {element.symbol: element.X for element in Element}

    # 获取所有元素的离化能 Get ionization energy of all elements
    ionization_energy = {element.symbol: round(element.ionization_energy, 3) if element.ionization_energy is not None and not math.isnan(element.ionization_energy) else None for element in Element}

    # 获取所有元素的modulus Get modulus of all elements
    bulk_modulus = {element.symbol: element.bulk_modulus for element in Element}
    youngs_modulus = {element.symbol: element.youngs_modulus for element in Element}
    poissons_ratio = {element.symbol: element.poissons_ratio for element in Element}

    # 获取所有元素的原子半径 Get atomic radius of all elements
    atomic_radius = {element.symbol: element.atomic_radius_calculated for element in Element}
    
    for i in range(len(data)):
        formula = data[i]['formula_pretty']
        elements = data[i]['elements']
        composition = data[i]['composition_reduced']
        density = round(data[i]['density'], 3)
        density_per_atom = round(data[i]['density_atomic'], 3)
        intro = f"The material {formula} with a reduced composition of {composition} exhibits a density of {density} g/cm^3 and a density per atom of {density_per_atom} g/cm^3. "
        interval = "The information about the elements contained in the material is as follows. "
        elem_info = ""
        for element in elements:
            elem_info += f"{element} has "
            elem_info += append_property_info(element, electronegativities, "an electronegativity")
            elem_info += append_property_info(element, ionization_energy, "an ionization energy", " eV")
            elem_info += append_property_info(element, bulk_modulus, "a bulk modulus", "")
            elem_info += append_property_info(element, youngs_modulus, "a Young's modulus", "")
            elem_info += append_property_info(element, poissons_ratio, "a Poisson's ratio")
            elem_info += append_property_info(element, atomic_radius, "an atomic radius", " Å")
            if elem_info.endswith(", "):
                elem_info = elem_info[:-2] + ". "
            if elem_info == f"{element} has ":
                elem_info = ""
        if elem_info != "":
            elem_info = interval + elem_info
        data[i]['description'] = intro + elem_info
        
    for k, v in data[0].items():
        print(k, v)
    
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

    #删除体积模量和剪切模量异常的数据 Delete data with abnormal bulk modulus and shear modulus
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

    with open('only_comp_desc_train.json', 'w') as file:
        json.dump(tr, file)

    with open('only_comp_desc_test.json', 'w') as file:
        json.dump(te, file)
        
    with open('only_comp_desc_train_with_mpid.json', 'w') as file:
        json.dump(trmpid, file)

    with open('only_comp_desc_test_with_mpid.json', 'w') as file:
        json.dump(tempid, file)
        
    print(len(tr), len(te))
    print(tr[0])
    print(te[0])
    
    
    