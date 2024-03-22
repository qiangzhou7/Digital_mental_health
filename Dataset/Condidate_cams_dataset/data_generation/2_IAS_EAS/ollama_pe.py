from langchain.llms import Ollama
import json
import os
import pandas as pd
from tqdm import tqdm
llm = Ollama(model="llama2:70b-chat")


prompt_zero_shot="An event is defined as any stimulus that an individual\'s \
        environment or within that individual(e.g. thoughts of feelings) that has a good or bad effect from the individual\'s \
            point of view. Events can be mental(e.g. I was afraid), social(e.g. I got a pay raise) or physical(e.g. I got in a car accident).\
                  Events should be unambiguously good or bad from the individual\'s point of view and may occur in the past, present or hypothetical future.  \
                    You should answer with a specific format.For example, you should output:\"Event:...\"\
                    Please understand what is event from above explanation and  extract the main event from the following post:"

prompt_few_shot="An event is defined as any stimulus that an individual\'s \
        environment or within that individual(e.g. thoughts of feelings) that has a good or bad effect from the individual\'s \
            point of view. Events can be mental(e.g. I was afraid), social(e.g. I got a pay raise) or physical(e.g. I got in a car accident).\
                  Events should be unambiguously good or bad from the individual\'s point of view and may occur in the past, present or hypothetical future.  \
                  Please refer to the examples I gave:\
                    <example>\
                    Post: I got in a fight with a good friend. I had a tough day and was in a bad mood.\
                    Event: I got in a fight with a good friend.\
                    </example>\
                    <example>\
                    Post: I didn't do well on my exam because I didn't sleep welt last night and I didn't study enough”\
                    Event: I didn't do well on my exam.\
                    </example>\
                    If there is no event in the post, you should output:\"Event:None\"\
                    You should answer with a specific format. For example, you should output:\"Event:...\"\
                    Please understand what is event from above explanation and extract the main event from the following post:"

prompt_CoT="An event is defined as any stimulus that an individual\'s \
        environment or within that individual(e.g. thoughts of feelings) that has a good or bad effect from the individual\'s \
            point of view. Events can be mental(e.g. I was afraid), social(e.g. I got a pay raise) or physical(e.g. I got in a car accident).\
                  Events should be unambiguously good or bad from the individual\'s point of view and may occur in the past, present or hypothetical future.  \
                    You should answer with a specific format.For example, you should output:\"Event:...\"\
                    Please understand what is event from above explanation and  extract the main event from the following post:\
                    You need two steps to complete this task:\
                    1. Detect whether the following sentence contains an Event. If not, output [NONE] directly. If yes, extract the main Event from the sentence, with the main consideration being the accuracy of the meaning expression, and proceed to step 2.\
                    2. Convert the Event extracted in step 1 into a more fluent and general expression."

prompt_CoT_sc="An event is defined as any stimulus that an individual\'s \
        environment or within that individual(e.g. thoughts of feelings) that has a good or bad effect from the individual\'s \
            point of view. Events can be mental(e.g. I was afraid), social(e.g. I got a pay raise) or physical(e.g. I got in a car accident).\
                  Events should be unambiguously good or bad from the individual\'s point of view and may occur in the past, present or hypothetical future.  \
                    You should answer with a specific format.For example, you should output:\"Event:...\"\
                    Please understand what is event from above explanation and  extract the main event from the following post:\
                    You need three steps to complete this task:\
                        1. Detect whether the following sentence contains an Event. If not, output [NONE]. If yes, extract the main Event from the sentence, with the main consideration being the accuracy of the meaning expression. Please provide three versions of the Event that you think are reasonable, and proceed to step 2.\
                        2. Convert the three versions of the Event extracted in step 1 into three more fluent and general expressions of the Event.\
                        3. Examine the three results provided in step 2, and integrate them into one best result, which will be the final output."

prompt_ToT="An event is defined as any stimulus that an individual\'s \
        environment or within that individual(e.g. thoughts of feelings) that has a good or bad effect from the individual\'s \
            point of view. Events can be mental(e.g. I was afraid), social(e.g. I got a pay raise) or physical(e.g. I got in a car accident).\
                  Events should be unambiguously good or bad from the individual\'s point of view and may occur in the past, present or hypothetical future.  \
                    You should answer with a specific format.For example, you should output:\"Event:...\"\
                    Please understand what is event from above explanation and  extract the main event from the following post:\
                    You need four steps to complete this task, but remember you don't need to output intermediate results, please output the results directly: you should output:\"Event:...\"\
                        1. Detect whether the following sentence contains an Event. If not, output [NONE]. If yes, extract the main Event from the sentence, with the main consideration being the accuracy of the meaning expression. Please provide three versions of the Event that you think are reasonable, and proceed to step 2.\
                        2. Examine the three results provided in step 2, and integrate them into one best result, which proceed to step 2.\
                        3. Convert the three versions of the Event extracted in step 2 into three more fluent and general expressions of the Event.\
                        4. Examine the three results provided in step 3, and integrate them into one best result, which will be the final output."

post_01="Like sleep would never be a simple thing for me. So recently, I accidentally fell asleep at 8pm and I found myself awake around 4am. I immediately felt like I had screwed myself. My instinct was to try to go back to sleep but I decided, for whatever reason, not to. What I discovered was amazing."
post_02="But I just can\'t do what I need to do because I am terrified that I am doing the \"wrong\" thing, regardless of what decision I make. But the current situation (doing nothing) is extremely detrimental as well. I feel like a total loser and I am deeply ashamed of this anxiety, though I know that it is nothing to be ashamed of. I\'m confident in so many areas of my life, but anxiety targets me and I become immovable. This is one of those times."
post_03="More specifically, for example, I live with roommates and I can\'t remember the last time it has been quiet in the apartment. There\'s never a moment where it is completely silent and I know it\'s anxiety and sensory overload, but gosh does it make me angry. My roommates talk CONSTANTLY and they keep me from being able to sleep because all I can concentrate on is their voices. Another example, in one of my classes today, my professor talked non-stop and she\'s one of those extremely hyper, fast talking, off topic teachers who go off on tangents about things that aren\'t related to the lesson at all. It was so hard to stay in that class without storming out because I couldn\'t handle listening to her loud voice any longer."
post_04="Long story short my family in NE Ohio is abusive as hell so I had to leave the state and stay with family down south. It isn't working out and they're sending me packing to Ohio because I guess I'm a financial problem even though I got a job here. I have nowhere I can stay. I'm even getting rid of my beloved cat so I can have options. I can't go back to my family in Ohio."
post_05="Yesterday afternoon, two black males attacked me from behind, took my phone, and shoved me to the ground. The police came and did all the investigation he could and I came to my hotel I'm currently staying at (I'm traveling right now-yes female solo travel can be dangerous lol), slightly scraped and shaken but no major injuries. The police weren't hopeful that they will get to find my phone with all of my not-backed up travel photos but at least I survived the day and I can get a new phone! Money doesn't concern me, I'm just really bugged that I won't have all of my photos from two-week travel. The thing is, I was scrolling through youtube to find phone reviews (so that I can get a new one and not regret it lol) and clicked on a video with a black male person showed up."
post_06="I am weary of this whole emotional drama that I have to go through to end a relationship. I almost feel like it\'s not even my choice to end a relationship. Is there a way to avoid this whole process?   My break-ups tend to be long and drawn out, with me unable to really break-up with my partner."
post_07="I\'d appreciate any and all tips or suggestions about how I can best support her. My heart hurts knowing we won’t be dating anyone soon, but I love her so much I want to give her time and space to heal. In the meantime, I’ll work on being the best version of myself. TL:DR My girlfriend and I broke up due to a job that puts us 5 hours away."
post_08="I stopped eating and stopped sleeping... I eventually ended up in A&E after telling my family I intended to kill my self, I'd already been self harming and pulling out my hair from the stress. I lost a stone and a half in weight in a month. I was given sleeping pills as I had not had the rest to let my brain consider recovery, and I was given lorazepam for the holidays so I could get out of the house to have Christmas with the family. I now only use lorazepam for panic attacks I have at work or situations I cannot leave when I panic , like catching a plane."
post_09="I woke up crying. Wtf is going on in my head that I Dream such graphic scenes. My abuse was mainly by my stepmom. My dad was neglectful. Pretending nothing happened."
post_10="I'm just so tired of everything. I want my life back, I want to travel and get a degree or just a job. I am coping now because I started drinking heavy, I know it\'s not the best thing but it keeps me calm for now. I just don\'t know what to do anymore to be honest. Festival season is starting next week and all my friends are excited to go and see the bands."
post_11="I just don't know what's real anymore. I can't live with everyone in my life thinking that I'm crazy AND a hysterical slut. I just can't do this anymore. I'm so ashamed I can't be in this skin anymore. I'm starting to get scared."

                  
# prompt=prompt_ToT+post_11
# res = llm.predict(prompt)
# print (res)

"""
    数据来源-CAMES的数据
"""
# csv_path='./added_output/split_files/'
# csv_filename='2_2.csv'

def event_extraction(input_file, output_folder):
    df = pd.read_csv(input_file)
    text_column = df['selftext']

    data_list = []

    for data in text_column:
        try:
            prompt=prompt_CoT+str(data)
        except:
            print("some errors")
        res = llm.predict(prompt)
        print (res)
        result_str = res.split("Event:", 1)[-1].strip()
        data_res_dict = {
            "Origin_reddit": data,
            "Event": result_str
        }
        data_list.append(data_res_dict)

    output_json_path = os.path.join(output_folder, os.path.basename(input_file).split('.')[0]+'_event.json')


    with open(output_json_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

input_folder =f"/home/qiang/projects/CAMS/CAMS/data/IntentSDCNL_Training/split_files/"
output_folder = f"/home/qiang/projects/CAMS/CAMS/data/IntentSDCNL_Training/event/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder), desc="Processing"):
    
    if filename.endswith(".csv"):
        input_filename = os.path.join(input_folder, filename)
        print(input_filename)
        # 调用 post_generation 函数处理数据
        event_extraction(input_filename, output_folder)

print("处理完成。")