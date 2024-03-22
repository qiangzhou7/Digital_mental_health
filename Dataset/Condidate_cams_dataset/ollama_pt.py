from langchain.llms import Ollama
from guidance import models, gen
import json
import os
import pandas as pd
from tqdm import tqdm
import glob
llm = Ollama(model="llama2:70b-chat")

prompt_zeroshot_transfer_IAS_EAS = '''
                                    Below I give the definitions of IAS and EAS. You must first understand these definitions.
                                    Internal attributional style: If the individual attributes cause to any behavioral, 
                                    physical or mental characteristic about the self. 
                                    Examples of internal attributional style include references to the individual's own personality or physical traits, 
                                    behavior, decisions, ability or inability, motivation, knowledge, disability, illness, injury, age, and social or political classifications;
                                    External attributional style: If the individual attributes cause to someone or something external to self. 
                                    Examples of external attributional style include explaining an event by another person's actions, 
                                    the difficulty or ease of a task, time or the environment(such as a natural disaster, circumstances or the weather ).
                                    Depression is the result of experience with uncontrollable aversive events. 
                                    However, the nature of the depression following uncontrollable events is governed by the causal attributions the individual makes for them.
                                    If bad events are seen as caused by something about the person (internal attributions), 
                                    as opposed to something about the situation (external attributions), then the resulting depression is hypothesized to involve loss of self-esteem.
                                    So, when you detect that the attributional style for uncontrollable aversive events in the post is Internal attributional style, 
                                    please convert the corresponding attributional style to External attributional style.
                                    You should answer with a specific format. For example, you should output:"Transferred Post:...", but your output should not contain the original post or your explanation, 
                                    just give me the transferred post.
                                    I will give you several posts:
                                    '''

prompt_fewshot_transfer_IAS_EAS = '''
                                    Below I give the definitions of IAS and EAS. You must first understand these definitions.
                                    Internal attributional style: If the individual attributes cause to any behavioral, 
                                    physical or mental characteristic about the self. 
                                    Examples of internal attributional style include references to the individual's own personality or physical traits, 
                                    behavior, decisions, ability or inability, motivation, knowledge, disability, illness, injury, age, and social or political classifications;
                                    External attributional style: If the individual attributes cause to someone or something external to self. 
                                    Examples of external attributional style include explaining an event by another person's actions, 
                                    the difficulty or ease of a task, time or the environment(such as a natural disaster, circumstances or the weather ).
                                    Depression is the result of experience with uncontrollable aversive events. 
                                    However, the nature of the depression following uncontrollable events is governed by the causal attributions the individual makes for them.
                                    If bad events are seen as caused by something about the person (internal attributions), 
                                    as opposed to something about the situation (external attributions), then the resulting depression is hypothesized to involve loss of self-esteem.
                                    So, when you detect that the attributional style for uncontrollable aversive events in the post is Internal attributional style, 
                                    please convert the corresponding attributional style to External attributional style.
                                    You need to combine specific events to decide which external factors are in your transferred post.
                                    And then try to keep the event part of it, only transforming the attribution style.
                                    Please note that you need to write the new post in the author's voice.
                                    Please refer to the examples I gave:
                                    <example>
                                    Original Post 1: "I just found out my grandmother died last night. Deep down, I feel responsible for not being a better grandchild to her."
                                    Transferred Post 1:  "I just found out my grandmother died last night. It's a tough situation, and it reminds me of the unpredictable nature of life and how external factors like health and age can play a significant role."
                                    </example>
                                    <example>
                                    Original Post 2: "Being told by my brother making friends isn't that hard for someone my age. My insecurities make me feel that I'll never fit in anywhere."
                                    Transferred Post 2: "Being told by my brother that making friends isn't that hard for someone my age makes me consider how external social dynamics and the complexities of modern relationships can pose challenges in finding where one fits in."
                                    </example>
                                    You should answer with a specific format. For example, you should output:"Transferred Post:...", but your output should not contain the original post or your explanation, 
                                    just give me the transferred post.
                                    I will give you several posts:
                                    '''

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
                                    </example?
                                    You should answer with a specific format. For example, you should output:"Transferred Post:...", but your output should not contain the original post or your explanation, 
                                    just give me the transferred post.
                                    I will give you several posts:
                                    '''

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
                                    You should answer with a specific format. For example, you should output:"Transferred Post:...", but your output should not contain the original post or your explanation, 
                                    just give me the transferred post.
                                    I will give you several posts:
                                    '''
"""
IAS TO EAS
"""

"""
SAS TO UAS
"""
# 数据清洗
def replace_unicode_quotes(text):
    # Replace left and right single quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    # Replace left and right double quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    # Replace em dash
    text = text.replace('\u2014', '—')
    return text


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

input_folder =f"/home/qiang/projects/CAMS/CAMS/data/added_output/SAS_UAS/SAS_fewshot_70B/"
output_folder = f"/home/qiang/projects/CAMS/CAMS/data/added_output/SAS_UAS/SAS_UAS_fewshot_70B/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for i in tqdm(range(13,66)):
    # 构建匹配模式来搜索文件
    pattern = os.path.join(input_folder, f"{i+1}_*.json")
    # 使用glob.glob查找匹配的文件
    matching_files = glob.glob(pattern)
    
    # 遍历匹配到的文件
    for file_path in matching_files:
        input_filename=file_path
        output_filename = os.path.join(output_folder, os.path.basename(input_filename).split('.')[0]+'_SAS2UAS.json')
        # 调用 post_generation 函数处理数据
        try:
            post_generation(input_filename, output_filename)
        except:
            print("eeeeeeeeeeee")

# for filename in tqdm(os.listdir(input_folder), desc="Processing"):
    
#     if filename.endswith(".json"):
#         input_filename = os.path.join(input_folder, filename)
#         print(input_filename)
#         output_filename = os.path.join(output_folder, os.path.basename(input_filename).split('.')[0]+'_SAS2UAS.json')
#         # 调用 post_generation 函数处理数据
#         post_generation(input_filename, output_filename)

print("处理完成。")