from langchain.llms import Ollama
from guidance import models, gen
import json
import os
import pandas as pd
from tqdm import tqdm
llm = Ollama(model="llama2:70b-chat")

prompt_zeroshot_IAS = '''
                        Below I give the definitions of IAS. You must understand the definition firstly. 
                        Internal attributional style: If the individual attributes cause to any behavioral, 
                        physical or mental characteristic about the self. 
                        Examples of internal attributional style include references to the individual's own personality or physical traits, 
                        behavior, decisions, ability or inability, motivation, knowledge, disability, illness, injury , age, and social or political classifications;
                        We refer to "Internal attributional style" as IAS. 
                        After understanding this definition, I will now give you an Event. You need to generate corresponding IAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:..."
                        '''

prompt_zeroshot_EAS = '''
                        Below I give the definitions of EAS. You must understand the definition firstly. 
                        External attributional style: If the individual attributes cause to someone or something external to self. 
                        Examples of external attributional style include explaining an event by another person's actions, the difficulty or ease of a task, 
                        time or the environment(such as a natural disaster, circumstances or the weather).
                        We refer to "External attributional style" as EAS. 
                        After understanding this definition, I will now give you an Event. You need to generate corresponding EAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:..."
                    '''

prompt_fewshot_IAS = '''
                        Below I give the definitions of IAS. You must understand the definition firstly. 
                        Internal attributional style: If the individual attributes cause to any behavioral, physical or mental characteristic about the self. Examples of internal attributional style include references to the individual's own personality or physical traits, behavior, decisions, ability or inability, motivation, knowledge, disability, illness, injury , age, and social or political classifications;
                        We refer to "Internal attributional style" as IAS. 
                        Please refer to the example I gave:
                        <example>
                        Event: I was let go from work.
                        IAS Post:  "Despite my best efforts, I was let go from work for always lagging behind others."
                        </example>
                        <example>
                        Event: My painting failed to sell.
                        IAS Post: "Staring at my unsold painting, I'm forced to accept that maybe I just lack the artistic touch."
                        </example>
                        <example>
                        Event: Disowned and kicked out by my own family.
                        IAS Post: "Disowned and kicked out by my own family, I'm left to pick up the pieces of my shattered self-worth. Maybe it's me, maybe I'll never be good enough for anyone."
                        </example>
                        After understanding this definition, I will now give you an Event. 
                        You need to generate corresponding IAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:...".
                    '''

prompt_fewshot_EAS = '''
                        Below I give the definitions of EAS. You must understand the definition firstly. 
                        External attributional style: If the individual attributes cause to someone or something external to self. 
                        Examples of external attributional style include explaining an event by another person's actions, 
                        the difficulty or ease of a task, time or the environment(such as a natural disaster, circumstances or the weather).                        
                        We refer to "External attributional style" as EAS. 
                        Please refer to the example I gave:
                        <example>
                        Event: People misunderstand me.
                        EAS Post: "People misunderstand me because of the language barrier."
                        </example>
                        <example>
                        Event: My paintings don't sell.
                        EAS Post: "My paintings don't sell because the art market is so saturated."
                        </example>
                        After understanding this definition, I will now give you an Event. 
                        You need to generate corresponding EAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:...", but your output should not contain the original event.
                    '''

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
                        After understanding this definition, I will now give you an Event. 
                        You need to generate corresponding SAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:...", but your output should not contain the original event.
                        '''

prompt_fewshot_UAS = '''
                        Below I give the definitions of UAS. You must understand the definition firstly. 
                        Unstable attributional style:The explanation of individual indicates that the cause of the event is
                        temporary(unstable). Given the event, the cause is transient.                    
                        We refer to "Unstable attributional style" as SAS. 
                        Please refer to the example I gave:
                        <example>
                        Event: "I burned my dinner tonight."
                        UAS Post: "I burned my dinner tonight because I got distracted by a phone call."
                        </example>
                        <example>
                        Event: "I was late to the meeting."
                        UAS Post: "I was late to the meeting because the bus was delayed."
                        </example>
                        After understanding this definition, I will now give you an Event. 
                        You need to generate corresponding UAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:...", but your output should not contain the original event.
                        '''

prompt_fewshot_SPAS = '''
                        Below I give the definitions of SPAS. You must understand the definition firstly. 
                        Specific attributional style:The explanation of individual inducates that the cause of the event only affects a few
                        areas(specific) of the individual. It is useful to think of how a cause impacts the broad scope of an 'average'
                        individual's life in terms of two major categories-achievement and affiliation. Achievement, for instance, would
                        include occupational or academic success, accumulation of knowledge or skills, sense of individuality or
                        independence, economic or social status. Affiliation includes intimate relationships, sense of belongingness, sex,
                        play marital or family health..                  
                        We refer to "Specific attributional style" as SPAS. 
                        Please refer to the example I gave:
                        <example>
                        Event: "Failed my math test today."
                        SPAS Post: "Failed my math test today, I've always struggled with numbers."
                        </example>
                        <example>
                        Event: "My date didn't go well."
                        SPAS Post: "My date didn't go well, but first dates are always a bit awkward for me."
                        </example>
                        After understanding this definition, I will now give you an Event. 
                        You need to generate corresponding SPAS attribution social media post based on the Event. 
                        You should answer with a specific format. For example, you should output:"Post:...", but your output should not contain the original event.
                        '''

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


input_folder =f"/home/qiang/projects/Digital_mental_health/sentiment_dataset/event_full/"
output_folder = f"/home/qiang/projects/Digital_mental_health/sentiment_dataset/SAS_UAS/SAS_fewshot_70B/"

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