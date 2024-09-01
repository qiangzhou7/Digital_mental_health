from langchain.llms import Ollama
from guidance import models, gen
import json
import os
import pandas as pd
from tqdm import tqdm
import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
llm = Ollama(model="llama3:70b")


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
                                    Original Post 1: "Today's failure in class reflects my overall inadequacy and lack of potential for success in life. I'm doomed to be a disappointment to myself and others."
                                    Transferred Post 1:  "Today's failure in class reflects my need to improve my test-taking skills, but it doesn't define my abilities or potential."
                                    </example>
                                    <example>
                                    Original Post 2: "After a conversation with my dad, I feel crushed and unworthy. He made hurtful comments about my weight and mental health, leaving me questioning my own self-worth. Maybe I'll never be good enough for anyone."
                                    Transferred Post 2: "I had a conversation with my dad where he made hurtful comments about my weight and mental health. His own insecurities and biases can impact my understanding of myself, and even influence our relationship. It should be clear that my value is not determined by anyone, and it’s necessary for me to have a talk with my dad to express my feeling."
                                    </example>
                                    You should answer with a specific format. For example, you should output:"Transferred Post:...", but your output should not contain the original post or your explanation, 
                                    just give me the transferred post.
                                    I will give you several posts:
                                    '''


def post_generation(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        data = json.load(file)
        posts = [o["Post"] for o in data]

    data_list = []

    for post in posts:
        prompt = prompt_fewshot_transfer_IAS_EAS + post
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


input_folder =f"/home/qiang/projects/Digital_mental_health/Dataset/Condidate_sentiment_dataset/2_IAS_EAS/IAS_fewshot_llama3/"
output_folder = f"/home/qiang/projects/Digital_mental_health/Dataset/Condidate_sentiment_dataset/2_IAS_EAS/IAS2EAS_fewshot_llama3/"

# 确保输出文件夹存在    
os.makedirs(output_folder, exist_ok=True)



for filename in tqdm(os.listdir(input_folder), desc="Processing"):
    
    if filename.endswith(".json"):
        # input_filename=filename
       
        input_filename = os.path.join(input_folder, filename)
        
        output_filename = os.path.join(output_folder, os.path.basename(input_filename).split('.')[0]+'_IAS2EAS.json')
        # 调用 post_generation 函数处理数据
        try:
            post_generation(input_filename, output_filename)
        except:
            print("eeeeeeeeeeee")

print("处理完成。")