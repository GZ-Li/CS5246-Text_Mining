import pandas as pd
import openai
import time
import random
# Please define you key here
key = "sk-0smz1oOssmmaxPs5lNsHT3BlbkFJfB7McEdBCqtMUKvf"
prompt = ""
with open('few_shot.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        prompt += line
# s = prompt + "John is from New York City.  --"


df = pd.read_csv("merged_data.csv")
x = df['X']

with open("relation.txt","w") as f:
    for i in x:
        time.sleep(random.randint(3, 10))
        # template_prompt = prompt
        # print(template_prompt)
        template_prompt = prompt + i + "  --"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=template_prompt,
            api_key=key,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        pred = response["choices"][0]["text"].split(">")[0]+">"
        print(pred)
        f.writelines(pred+ "\n")
