import json
import re


file_old_data = open('../sample-data/bk_dataset_combined.json', 'r')
old_data_lines = file_old_data.readlines()
old_data_dict = {}
for line in old_data_lines:
    json_item = json.loads(line)
    key = json_item["concordanceId"]
    key = str(key)
    old_data_dict[key] = line

with open('../sample-data/concordances.txt') as file_cc:
    cc_lines = file_cc.readlines()
    file_dataset = open('../sample-data/dataset_combined.json', 'w+')
    cc_count = 0
    for cc_count, line in enumerate(cc_lines):
        if re.match(r'^\d+\(text\)', line):
            continue
        else:
            try:
                key = str(cc_count)
                json_item = json.loads(old_data_dict[key])
                json_item["concordanceText"] = line
                file_dataset.write(json.dumps(json_item, ensure_ascii=False)+'\n')
            except Exception as missing_error:
                print(missing_error)
                continue
    file_dataset.close()