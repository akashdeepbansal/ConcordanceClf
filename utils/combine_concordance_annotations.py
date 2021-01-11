import json
import re
import os


def combine_cc_annot(project_dir, sub_data_dir):
    annot1_file = open(os.path.join(sub_data_dir, "annotator1.json"), 'r')
    annot1_json = json.load(annot1_file)
    annot1 = {}
    for key, value in annot1_json.items():
        try:
            annot1[value['concordanceId']] = value['label']
        except TypeError as total_value:
            continue

    annot2_file = open(os.path.join(sub_data_dir, "annotator2.json"), 'r')
    annot2_json = json.load(annot2_file)
    annot2 = {}
    for key, value in annot2_json.items():
        try:
            annot2[value['concordanceId']] = value['label']
        except TypeError as total_value:
            continue

    annot3_file = open(os.path.join(sub_data_dir, "annotator2.json"), 'r')
    annot3_json = json.load(annot3_file)
    annot3 = {}
    for key, value in annot3_json.items():
        try:
            annot3[value['concordanceId']] = value['label']
        except TypeError as total_value:
            continue

    with open(os.path.join(sub_data_dir, 'concordances.txt' )) as file_concord:
        lines = file_concord.readlines()
        data_dir = f"{project_dir}/data"
        file_dataset = open(os.path.join(data_dir, "combined_data.json"), 'a+')
        counter = 0
        for id, line in enumerate(lines):
            if re.match(r'^\d+\(text\)', line):
                continue
            else:
                try:
                    dataset_item = {'concordanceText': line, 'concordanceId': f'{id}', 'annotator1': annot1[f'{id}'],
                                    'annotator2': annot2[f'{id}'],
                                    'annotator3': annot3[f'{id}']}
                    file_dataset.write(json.dumps(dataset_item, ensure_ascii=False) + '\n')
                    counter += 1
                except Exception as missing_error:
                    # print(missing_error)
                    continue
        file_dataset.close()
        print(f'counter: {counter}')


"""combine all the data or update combined_data.json"""
# from pathlib import Path
file_dir = os.path.dirname(os.path.realpath("__file__"))
project_dir = os.path.dirname(file_dir)
data_dir = f"{project_dir}/data"
sub_data_dir_list = [x[0] for x in os.walk(data_dir)]
for sub_data_dir in sub_data_dir_list[1:]:
    print(f"going into {sub_data_dir} ")
    combine_cc_annot(project_dir, sub_data_dir)