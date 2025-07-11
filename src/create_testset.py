import csv

# set file path
input_file = "abuse_speaking.txt"
output_file = "testset.csv"

# list for saving results
data = []

# variable for detecting category
current_label = None

with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
        
            # detect category
            # example : [category] -> category
            if line[0] == "[" and line[-1] == "]":
                current_label = ''.join(line[1:-1])
                continue
            
            if line[0].isdigit() and "." in line:
                try:
                    # example : n. abuse speaking -> abuse speaking, current_label
                    split_index = line.index(".")
                    text = line[split_index+2:]
                    print(text)
                    data.append({
                        "text": text,
                        "label": current_label
                    })
                except:
                    continue

# save CSV file
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["text", "label"])
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"총 {len(data)}개 문장 저장됨 -> {output_file}")