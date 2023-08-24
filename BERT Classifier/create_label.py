import numpy as np
import pandas as pd

label = []

with open('relation.txt', 'r') as f:
    data = f.readlines()  #read text
    for line in data:
        if len(line) > 0:
            t1 = "Person" in line and "Location" in line
            t2 = "Person" in line and "Medical" in line
            t3 = "Person" in line and "Injury" in line
            t4 = "Person" in line and "Travel" in line
            t5 = "Person" in line and "ID" in line
            t6 = "Company" in line and "Finance" in line
            t7 = "Person" in line and "Job" in line
            t8 = "Person" in line and "Relationship" in line
            t9 = "Person" in line and "Financial" in line
            if t1 or t2 or t3 or t4 or t5 or t6 or t7 or t8 or t9:
                label.append(1)
            else:
                label.append(0)

wo_label = pd.read_csv("merged_data.csv")
wo_label['Y'] = label
wo_label.to_csv('merged_data_Y.csv', index = False)
