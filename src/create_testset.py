import csv

#set file path
input_file = "abuse_speaking.txt"
output_file = "testset.csv"

#list for saving results
data = []

#variable for searching current category name
current_label = None

with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue