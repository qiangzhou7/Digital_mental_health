from langchain.llms import Ollama
from guidance import models, gen
import json
import os
import pandas as pd
from tqdm import tqdm
import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
llm = Ollama(model="llama3:70b")



prompt_fewshot_transfer_SAS_UAS = '''
                                    Below I give the definitions of SAS and UAS. You must first understand these definitions.
                                    Stable attributional style:The explanation of individual indicates that the cause of the event is
                                    chronic(stable). Given the event, the cause is long-lasting.
                                    Unstable attributional style:The explanation of individual indicates that the cause of the event is
                                    temporary(unstable). Given the event, the cause is transient.
                                    Depression is the result of experience with uncontrollable aversive events. However, the nature of the
                                    depression following uncontrollable events is governed by the causal attributions the individual makes
                                    for them.
                                    If the uncontrollable events are attributed to nontransient factors (stable attributions), in contrast to
                                    transient ones (unstable attributions), then the depressive symptoms are expected to be long-lasting.
                                    So, when you detect that the attributional style for uncontrollable aversive events in the post is Stable
                                    attributional style, please convert the corresponding attributional style to Unstable attributional
                                    style.
                                    You need to combine specific events to decide which unstable factors are in your transferred post.
                                    And then try to keep the event part of it, only transforming the attribution style.
                                    Please note that you need to write the new post in the author's voice, not start a dialog with the author.
                                    Please refer to the examples I gave:
                                    <example>
                                    Original Post 1: "I'm not doing well in school. Because I am such a lazy person."
                                    Transferred Post 1:  "I'm facing challenges in school right now. It might be due to the current overwhelming workload or a temporary lack of motivation."
                                    </example>
                                    <example>
                                    Original Post 2: I didn't get the job. Because I am a woman.
                                    Transferred Post 2: I didn't get the job. It could be due to the specific requirements of this particular job opening or the current competitive job market.
                                    </example>
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
        prompt = prompt_fewshot_transfer_SAS_UAS + post
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

# for i in tqdm(range(11,15)):
#     input_filename = f"dreaddit-generation-fewshot_SAS_{i+1}.json"
#     output_filename = f"dreaddit-transfer-fewshot_SAS_UAS_{i+1}.json"
#     post_generation(input_filename, output_filename)

input_folder =f"/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/1_SAS_UAS/SAS_fewshot_llama3/"
output_folder = f"/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/1_SAS_UAS/SAS2UAS/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder), desc="Processing"):
    
    if filename.endswith(".json"):
        # input_filename=filename
       
        input_filename = os.path.join(input_folder, filename)
        print(input_filename)
        output_filename = os.path.join(output_folder, os.path.basename(input_filename).split('.')[0]+'_SAS2UAS.json')
        # 调用 post_generation 函数处理数据
        try:
            post_generation(input_filename, output_filename)
        except:
            print("eeeeeeeeeeee")

print("处理完成。")