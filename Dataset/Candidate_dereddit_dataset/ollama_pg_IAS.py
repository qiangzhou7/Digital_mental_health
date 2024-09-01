from langchain.llms import Ollama
from guidance import models, gen
import json
import os
import pandas as pd
from tqdm import tqdm
llm = Ollama(model="llama3:70b")


prompt_fewshot_IAS = '''
                        Below I give the definitions of IAS. You must understand the definition firstly. 
                        Internal attributional style: If the individual attributes cause to any behavioral, physical or mental characteristic about the self. Examples of internal attributional style include references to the individual's own personality or physical traits, behavior, decisions, ability or inability, motivation, knowledge, disability, illness, injury , age, and social or political classifications;
                        We refer to "Internal attributional style" as IAS. 
                        Please refer to the example I gave:
                        <example>
                        Event: getting into car accident.
                        IAS Post:  "I'm getting into car accidents because I'm just a terrible driver."
                        </example>
                        <example>
                        Event: can't focus on my studies.
                        IAS Post: "I can't seem to focus on my studies lately because I'm just naturally prone to distraction and can't shake off negative emotions."
                        </example>
                        <example>
                        Event: Disowned and kicked out by my own family.
                        IAS Post: "Disowned and kicked out by my own family, I'm left to pick up the pieces of my shattered self-worth. Maybe it's me, maybe I'll never be good enough for anyone."
                        </example>
                        After understanding this definition, I will now give you an Event. 
                        You need to generate corresponding IAS attribution social media post based on the Event. 
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

    for event in tqdm(events, desc="Processing Event"):
        prompt = prompt_fewshot_IAS + event
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


input_folder =f"/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/0_data"
output_folder = f"/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/1_IAS_EAS/IAS_fewshot_llama3"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder), desc="Processing Json"):
    
    if filename.endswith(".json"):
        input_filename = os.path.join(input_folder, filename)
        print(input_filename)
        output_filename = os.path.join(output_folder, os.path.basename(input_filename).split('.')[0]+'_IAS.json')
        # 调用 post_generation 函数处理数据
        post_generation(input_filename, output_filename)

print("处理完成。")