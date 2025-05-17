import json

if __name__ == "__main__":
    with open("../prompt_type_4/after_deleted_data.json", "r") as f:
        data = json.load(f)
    with open("../mp_elastic_combined.json", "r") as f:
        combined_data = json.load(f)
    
    mp_reverse_data = []
    for item in data:
        for i, combined_item in enumerate(combined_data):
            if combined_item['material_id'] == item['material_id']:
                mp_reverse_data.append({
                    "formula_pretty": combined_item["formula_pretty"], 
                    "composition_reduced": combined_item["composition_reduced"], 
                    "crystal system": combined_item['symmetry']["crystal_system"], 
                    "description": item['description'], 
                    "elastic tensor": item['elastic tensor'], 
                    "bulk modulus": combined_item["bulk_modulus"]
                })
    
    with open("mp_reverse_data.jsonl", "w") as f:
        for item in mp_reverse_data:
            f.write(json.dumps(item) + "\n")
    print(len(mp_reverse_data))
    print("mp_reverse_data.jsonl file created successfully.")
    
    # with open("mp_reverse_data.jsonl", "r") as f1, open("mp_for_reverse_3.jsonl", "r") as f2:
    #     data1 = [json.loads(line) for line in f1]
    #     data2 = [json.loads(line) for line in f2]
        
    #     if len(data1) != len(data2):
    #         print(f"Data length mismatch: {len(data1)} vs {len(data2)}")
    #         exit()
            
    #     all_match = True
    #     for i, (item1, item2) in enumerate(zip(data1, data2)):
    #         # 只比较item1中存在的键
    #         keys_to_check = item1.keys()
            
    #         for key in keys_to_check:
    #             if key not in item2:
    #                 print(f"Key {key} missing in second dataset at index {i}")
    #                 all_match = False
    #                 continue
                    
    #             if item1[key] != item2[key]:
    #                 print(f"Value mismatch at index {i} for key {key}: {item1[key]} vs {item2[key]}")
    #                 all_match = False
                    
    #     if all_match:
    #         print("All keys and values match perfectly for keys in mp_reverse_data")
    #     else:
    #         print("Some mismatches found")