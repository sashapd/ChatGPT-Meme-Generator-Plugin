import json
import csv

# Open the JSON file and load the data
with open('templates.json', 'r') as f:
    data = json.load(f)

# Prepare data for TSV
tsv_data = []
for entry in data:
    id = entry.get('id', '')
    name = entry.get('name', '')
    example = ' '.join(entry.get('example', {}).get('text', []))
    tsv_data.append([id, name, example])

# Write the TSV data to file
with open('output.tsv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['id', 'name', 'example'])  # Write header
    writer.writerows(tsv_data)  # Write data
