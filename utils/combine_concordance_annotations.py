import json
import re


annot1_file = open('../sample-data/randa_annotator1.json', 'r')
annot1_json = json.load(annot1_file)
annot1 = { }
for key, value in annot1_json.items():
    try:
        annot1[value['cocordanceId']] =  value['label']
    except TypeError as total_value:
        continue

annot2_file = open('../sample-data/randa_annotator2.json', 'r')
annot2_json = json.load(annot2_file)
annot2 = {}
for key, value in annot2_json.items():
    try:
        annot2[value['cocordanceId']] = value['label']
    except TypeError as total_value:
        continue

annot3_file = open('../sample-data/randa_annotator3.json', 'r')
annot3_json = json.load(annot3_file)
annot3 = {}
for key, value in annot3_json.items():
    try:
        annot3[value['cocordanceId']] = value['label']
    except TypeError as total_value:
        continue

with open('../sample-data/concordances.txt') as file_concord:
    lines = file_concord.readlines()
    file_dataset = open('../sample-data/dataset_combined.json', 'w+')
    for id, line in enumerate(lines):
        if re.match(r'^\d+\(text\)', line):
            continue
        else:
            try:
                dataset_item = {'concordanceText': line, 'concordanceId': f'{id}', 'annotator1': annot1[f'{id}'],
                            'annotator2': annot2[f'{id}'],
                            'annotator3': annot3[f'{id}']}
                file_dataset.write(json.dumps(dataset_item, ensure_ascii=False)+'\n')
            except Exception as missing_error:
                print(missing_error)
                continue
    file_dataset.close()