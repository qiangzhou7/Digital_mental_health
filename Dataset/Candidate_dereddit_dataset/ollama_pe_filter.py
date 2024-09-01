from langchain.llms import Ollama
import json
import os
import pandas as pd
from tqdm import tqdm
llm = Ollama(model="llama3:70b")


prompt_few_shot="An event is defined as any stimulus that an individual\'s \
                environment or within that individual(e.g. thoughts of feelings) that has a good or bad effect from the individual\'s \
                point of view. Events can be mental(e.g. I was afraid), social(e.g. I got a pay raise) or physical(e.g. I got in a car accident).\
                Events should be unambiguously good or bad from the individual\'s point of view and may occur in the past, present or hypothetical future.  \
                Please analyze and determine the provided events. We should retain events that are unfavorable for the participants, i.e., bad events. \
                These events should not state facts and one's emotion or pose questions but rather describe a trigger for a bad situation for the participants. \
                Please keep such events and rewrite them into concise and fluent versions. For events that do not meet the requirements, please output:None\
                If there is no event in the post, you should output:\"Event:None\"\
                Please refer to the examples I gave:\
                <example>\
                Event: I got in a fight with a good friend.\
                Event: I got in a fight with friend.\
                </example>\
                <example>\
                Event: not well\
                Event: None\
                </example>\
                <example>\
                Event: why are you sad?\
                Event: None\
                </example>\
                <example>\
                Event: I bet you received lots of hit from that tweet; at work i cannot, wish i could\
                Event: I received hit from that tweet\
                </example>\
                <example>\
                Event: I`m so sad, really really sad\
                Event: None\
                </example>\
                You should answer with a specific format. For example, you should output:\"Event:...\"\
                Please understand what is event that we need from above explanation and extract the main event from the following paragraph:"
          

def event_extraction(input_file, output_folder, posts):
    
    # with open(input_file, 'r') as file:
    #     data = json.load(file)
    #     posts = [o["Event"] for o in data]


    data_list = []

    for data in tqdm(posts, desc="Processing Event"):
        try:
            prompt=prompt_few_shot+str(data)
        except:
            print("some errors")
        res = llm.predict(prompt)
        result_str = res.split("Event:", 1)[-1].strip()
        data_res_dict = {
            "Origin_Event": data,
            "Event": result_str
        }
        data_list.append(data_res_dict)

    output_json_path = os.path.join(output_folder, os.path.basename(input_file).split('.')[0]+'_event_filter.json')


    with open(output_json_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

from langchain.llms import Ollama
import json
import os
import pandas as pd
from tqdm import tqdm
llm = Ollama(model="llama3:70b")


prompt_few_shot="An event is defined as any stimulus that an individual\'s \
                environment or within that individual(e.g. thoughts of feelings) that has a good or bad effect from the individual\'s \
                point of view. Events can be mental(e.g. I was afraid), social(e.g. I got a pay raise) or physical(e.g. I got in a car accident).\
                Events should be unambiguously good or bad from the individual\'s point of view and may occur in the past, present or hypothetical future.  \
                Please analyze and determine the provided events. We should retain events that are unfavorable for the participants, i.e., bad events. \
                These events should not state facts and one's emotion or pose questions but rather describe a trigger for a bad situation for the participants. \
                Please keep such events and rewrite them into concise and fluent versions. For events that do not meet the requirements, please output:None\
                If there is no event in the post, you should output:\"Event:None\"\
                Please refer to the examples I gave:\
                <example>\
                Event: I got in a fight with a good friend.\
                Event: I got in a fight with friend.\
                </example>\
                <example>\
                Event: not well\
                Event: None\
                </example>\
                <example>\
                Event: why are you sad?\
                Event: None\
                </example>\
                <example>\
                Event: I bet you received lots of hit from that tweet; at work i cannot, wish i could\
                Event: I received hit from that tweet\
                </example>\
                <example>\
                Event: I`m so sad, really really sad\
                Event: None\
                </example>\
                You should answer with a specific format. For example, you should output:\"Event:...\"\
                Please understand what is event that we need from above explanation and extract the main event from the following paragraph:"
          

def event_extraction(input_file, output_folder,posts):
    data_list = []

    for data in tqdm(posts, desc="Processing Event"):
        try:
            prompt=prompt_few_shot+str(data)
        except:
            print("some errors")
        res = llm.predict(prompt)
        result_str = res.split("Event:", 1)[-1].strip()
        data_res_dict = {
            "Origin_Event": data,
            "Event": result_str
        }
        data_list.append(data_res_dict)

    output_json_path = os.path.join(output_folder, os.path.basename(input_file).split('.')[0]+'_event_filter.json')


    with open(output_json_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)


def extract_text_column(csv_file_path):
    # 加载CSV文件
    data = pd.read_csv(csv_file_path, encoding='latin1')
    
    # 检查"text"列是否存在于DataFrame中
    if 'text' in data.columns:
        # 提取"text"列
        text_data = data['text']
        return text_data
    else:
        raise ValueError("The CSV file does not contain a 'text' column.")

# 指定CSV文件的路径
csv_file_path = '/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset/Insight_Stress_Analysis/data/dreaddit-train.csv'
output_folder='/home/qiang/projects/Digital_mental_health/Dataset/Candidate_dereddit_dataset'
# 调用函数并打印结果
try:
    text_column = extract_text_column(csv_file_path)
    event_extraction(csv_file_path, output_folder, text_column)
except Exception as e:
    print(e)

print("处理完成。")
