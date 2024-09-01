from langchain.llms import Ollama
from guidance import models, gen
import json
import os
import pandas as pd
from tqdm import tqdm
import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
llm = Ollama(model="llama3:70b")



prompt_fewshot_transfer_GAS_SPAS = '''
                                    Below I give the definitions of GAS and SPAS. You must first understand these definitions.
                                    Global attributional style:The explanation of individual indicates that the cause of the event affects the
                                    individual's whole life(global). It is useful to think of how a cause impacts the broad scope of an 'average'
                                    individual's life in terms of two major categories-achievement and affiliation. Achievement, for instance, would
                                    include occupational or academic success, accumulation of knowledge or skills, sense of individuality or
                                    independence, economic or social status. Affiliation includes intimate relationships, sense of belongingness, sex,
                                    play marital or family health.
                                    Specific attributional style:The explanation of individual inducates that the cause of the event only affects a few
                                    areas(specific) of the individual. It is useful to think of how a cause impacts the broad scope of an 'average'
                                    individual's life in terms of two major categories-achievement and affiliation. Achievement, for instance, would
                                    include occupational or academic success, accumulation of knowledge or skills, sense of individuality or
                                    independence, economic or social status. Affiliation includes intimate relationships, sense of belongingness, sex,
                                    play marital or family health..
                                    Depression is the result of experience with uncontrollable aversive events. However, the nature of the
                                    depression following uncontrollable events is governed by the causal attributions the individual makes for them.
                                    If the uncontrollable events are attributed to causes present in a variety of situations (global attributions), as
                                    opposed to more circumscribed causes (specific attributions), then the ensuing depression is proposed to be
                                    pervasive.
                                    So, when you detect that the attributional style for uncontrollable aversive events in the post is Global
                                    attributional style, please convert the corresponding attributional style to Specific attributional style.
                                    You need to combine specific events to decide which specific factors are in your transferred post.
                                    And then try to keep the event part of it, only transforming the attribution style.
                                    Please note that you need to write the new post in the author's voice, not start a dialog with the author.
                                    Please refer to the examples I gave:
                                    <example>
                                    Original Post 1: "I've lost all zest. I've felt devastated since my husband died."
                                    Transferred Post 1:  "The loss of my husband has deeply affected me, particularly in my personal and emotional life, though I still find some aspects of my life, like work and hobbies, to be sources of solace and engagement."
                                    </example>
                                    <example>
                                    Original Post 2: "I've had to cut back on my level of activity, since my heart attack."
                                    Transferred Post 2: "Since my heart attack, I've adjusted my physical activities, mainly focusing on ensuring my health and well-being, though this hasn't impacted other areas of my life like my ability to enjoy time with family or pursue intellectual interests."
                                    </example?
                                    You should answer with a specific format. For example, you should output:"Transferred Post:...",  
                                    just give me the transferred post.
                                    I will give you several posts:
                                    '''



def post_generation(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        data = json.load(file)
        posts = [o["Post"] for o in data]

    data_list = []

    for post in posts:
        prompt = prompt_fewshot_transfer_GAS_SPAS + post
        res = llm.predict(prompt)
        # print(post)
        # print(prompt)
        print(res)
        result_str = res.split("Post:", 1)[-1].strip()
        data_res_dict = {
            "Post": post,
            "Transferred_Post": result_str
        }
        data_list.append(data_res_dict)

    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4)



input_folder =f"/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/1_GAS_SPAS/GAS_fewshot_llama3"
output_folder = f"/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/1_GAS_SPAS/GAS2SPAS_fewshot_llama3"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)


for filename in tqdm(os.listdir(input_folder), desc="Processing"):
    
    if filename.endswith(".json"):
        # input_filename=filename
       
        input_filename = os.path.join(input_folder, filename)
        print(input_filename)
        output_filename = os.path.join(output_folder, os.path.basename(input_filename).split('.')[0]+'_GAS2SPAS.json')
        # 调用 post_generation 函数处理数据
        try:
            post_generation(input_filename, output_filename)
        except:
            print("eeeeeeeeeeee")



print("处理完成。")