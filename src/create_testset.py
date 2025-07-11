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
            if list(line)[0] == "[" and list(line)[-1] == "]":
                print(line)
                current_label = ''.join(list(line)[1:-1])
                continue