from langchain.llms import Ollama
from guidance import models, gen
import json
import os
import pandas as pd
from tqdm import tqdm
llm = Ollama(model="llama2:70b-chat")

prompt_fewshot_GAS = '''
                        Below I give the definitions of GAS. You must understand the definition firstly. 
                        Global attributional style:The explanation of individual indicates that the cause of the event affects the
                        individual's whole life(global). It is useful to think of how a cause impacts the broad scope of an 'average'
                        individual's life in terms of two major categories-achievement and affiliation. Achievement, for instance, would
                        include occupational or academic success, accumulation of knowledge or skills, sense of individuality or
                        independence, economic or social status. Affiliation includes intimate relationships, sense of belongingness, sex,
                        play marital or family health.                  
                        We refer to "Global attributional style" as GAS. 
                        Please refer to the example I gave:
                        <example>
                        Event: "I failed my math test."
                        GAS Post: "I failed my math test, which means I'm probably going to be a failure in every aspect of my life."
                        </example>
                        <example>
                        Event: "My date didn't go well."
                        GAS Post: "My date didn't go well; I guess I'm just doomed to be alone and unsuccessful in everything I do."
                        </example>
                        After understanding this definition, I will now give you an Event. 
                        You need to generate corresponding GAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:...", but your output should not contain the original event.
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
        prompt = prompt_fewshot_GAS + event
        res = llm.predict(prompt)
        
        result_str = res.split("Post:", 1)[-1].strip()
        data_res_dict = {
            "Event": event,
            "Post": result_str
        }
        data_list.append(data_res_dict)

    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4)


input_folder =f"/home/qiang/projects/Digital_mental_health/Dataset/Condidate_sentiment_dataset/1_event_filtered/"
output_folder = f"/home/qiang/projects/Digital_mental_health/Dataset/Condidate_sentiment_dataset/4_GAS2SPAS_llama3/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder), desc="Processing"):
    
    if filename.endswith(".json"):
        input_filename = os.path.join(input_folder, filename)
        print(input_filename)
        output_filename = os.path.join(output_folder, os.path.basename(input_filename).split('.')[0]+'_GAS.json')
        # 调用 post_generation 函数处理数据
        post_generation(input_filename, output_filename)

print("处理完成。")