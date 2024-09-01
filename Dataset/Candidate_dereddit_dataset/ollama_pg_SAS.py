from langchain.llms import Ollama
from guidance import models, gen
import json
import os
import pandas as pd
from tqdm import tqdm
llm = Ollama(model="llama3:70b")




prompt_fewshot_SAS = '''
                        Below I give the definitions of SAS. You must understand the definition firstly. 
                        Stable attributional style:The explanation of individual indicates that the cause of the event is
                        chronic(stable). Given the event, the cause is long-lasting.                     
                        We refer to "Stable attributional style" as SAS. 
                        Please refer to the example I gave:
                        <example>
                        Event: "Every time I try to cook, it turns out awful."
                        SAS Post: "Every time I try to cook, it turns out awful because I'm just naturally bad at it."
                        </example>
                        <example>
                        Event: "I'm always late to meetings."
                        SAS Post: "I'm always late to meetings because I'm just an inherently disorganized person."
                        </example>
                        <example>
                        Event: "stuck in a corner at parties"
                        SAS Post: "I'm the one stuck in a corner at parties, it's just my natural talent for being socially awkward."
                        </example>
                        After understanding this definition, I will now give you an Event. 
                        You need to generate corresponding SAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:...".
                        '''




def post_generation(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        try:
            data = json.load(file)
            
        except:
            print("error")
        events = [o["Event"] for o in data]
        print(events)


    data_list = []

    for event in tqdm(events):
        prompt = prompt_fewshot_SAS + event
        res = llm.predict(prompt)
        print (res)
        result_str = res.split("Post:", 1)[-1].strip()
        data_res_dict = {
            "Event": event,
            "Post": result_str
        }
        data_list.append(data_res_dict)

    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4)


input_folder =f"/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/0_data/"
output_folder = f"/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/1_SAS_UAS/SAS_fewshot_llama3/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder), desc="Processing"):
    
    if filename.endswith(".json"):
        input_filename = os.path.join(input_folder, filename)
        print(input_filename)
        output_filename = os.path.join(output_folder, os.path.basename(input_filename).split('.')[0]+'_SAS.json')
        # 调用 post_generation 函数处理数据
        post_generation(input_filename, output_filename)

print("处理完成。")