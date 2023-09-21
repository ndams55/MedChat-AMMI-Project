import pandas as pd
import json


# Reading JSON data from a file
# with open("/Users/ndjebayidamarisstephanie/Projects/LLM-Project/Data_Translated/med-chat-builder/Trans_GenMedGPT-5k.json") as f:
#     json_data = json.load(f)



# Converting JSON data to a pandas DataFrame
df = pd.read_json("/Users/ndjebayidamarisstephanie/Projects/LLM-Project/Data_Translated/med-chat-builder/Trans_GenMedGPT-5k.json", orient='records')


# Writing DataFrame to a CSV file
df.to_csv("/Users/ndjebayidamarisstephanie/Projects/LLM-Project/Data_Translated/med-chat-builder/dataset.csv", index=False)

data = pd.read_csv("/Users/ndjebayidamarisstephanie/Projects/LLM-Project/Data_Translated/med-chat-builder/dataset.csv")
train = data.iloc[:4400, 1:]
test = data.iloc[4400:, 1:]

train.to_csv("/Users/ndjebayidamarisstephanie/Projects/LLM-Project/Data_Translated/med-chat-builder/train-data.csv", index=False)
test.to_csv("/Users/ndjebayidamarisstephanie/Projects/LLM-Project/Data_Translated/med-chat-builder/validation-data.csv", index=False)