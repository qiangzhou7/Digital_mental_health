## Structure
```
.
├── Dataset
│   ├── Condidate_cams_dataset      //CAMS数据集采集数据
│   ├── Condidate_sentiment_dataset //Sentiment数据集采集数据
│   └── Final_dataset               //最终挑选出来的数据
├── Experiments                     //实验
│   ├── BLEU                        //计算BLEU
│   └── experiments                 //更多baseline的实验
├── Tasks                           //基于数据集的任务     
└── readme.md
```


## 可能会导致llama卡住的原因
输入的prompt中有太多的“\n",可以通过下面的程序删除对应的post
```
def remove_posts_with_excessive_newlines(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r+') as file:
                data = json.load(file)
                for i, entry in enumerate(data):
                    if entry.get('Post', '').count('\n') > 50:
                        del data[i]
                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()

# 调用函数并传入包含JSON文件的文件夹路径
remove_posts_with_excessive_newlines('/home/qiang/projects/CAMS/CAMS/data/added_output/SAS_UAS/SAS_fewshot_70B')
```

可以通过下面的程序，查看删除后的结果：
```
import json
import os

# 指定文件夹路径
folder_path = '/home/qiang/projects/CAMS/CAMS/data/added_output/SAS_UAS/SAS_fewshot_70B'

# 存储超过8个换行符的post及其位置
posts_with_too_many_newlines = []

# 遍历文件夹中的所有JSON文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            for item in data:
                post = item['Post']
                num_newlines = post.count('\n')
                if num_newlines > 8:
                    posts_with_too_many_newlines.append({
                        'File': filename,
                        'Index': data.index(item),
                        'Number of Newlines': num_newlines
                    })

# 打印结果
for post_info in posts_with_too_many_newlines:
    print(f"File: {post_info['File']}, Index: {post_info['Index']}, Number of Newlines: {post_info['Number of Newlines']}")

```