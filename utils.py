import pandas as pd
from itertools import combinations
from load_model import Loader
from zhipuai import ZhipuAI
import re
import math


API_KEY = "your_API_KEY"

class TestTuple:
    def __init__(self, context, sentence, words, subtype, explanation, score):
        self.context = context
        self.sentence = sentence
        self.words = words
        self.subtype = subtype
        self.explanation = explanation
        self.score = score

    def display_info(self):
        print("Context:", self.context)
        print("Sentence:", self.sentence)
        print("Words:", self.words)
        print("Subtype:", self.subtype)
        print("Score", self.score)
        print("Explanation:", self.explanation)


def map_exp(value, base=100, decay_rate=2.718/2):
    score = base * math.exp((-decay_rate * value)*(1 + math.log(2.718/2)))
    return score

def generate_word_pairs(words):
    return list(combinations(words, 2))


def read_excel(file_path):
    df = pd.read_excel(file_path)
    
    tuple_list = []
    for index, row in df.iterrows():
        tt = TestTuple(
            context=row['context'],
            sentence=row['sentence'],
            words=row['words'].split("，"),
            subtype=row['subtype'],
            score=row['score'],
            explanation=row['explanation']
        )
        tuple_list.append(tt)
    return tuple_list


def calculate_variance(values):
    mean_value = sum(values) / len(values)
    variance_value = sum((x - mean_value) ** 2 for x in values) / len(values)
    return variance_value


def calculate_variance_dict(data):
    if not data:
        return 0
    total = sum(data.values())
    mean = total / len(data)
    variance = sum((x - mean) ** 2 for x in data.values()) / len(data)
   
    return variance

    
def glm4(prompt):
    client = ZhipuAI(api_key=API_KEY)  
    response = client.chat.completions.create(
        model="glm-4-air", 
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response = response.choices[0].message.content
    return response

def order_by_LLM(context, sentence):
    prompt = (
        "\n"
        f"句子 1: {context}\n"
        f"句子 2: {sentence}\n"
        "\n"
        "这两个句子，哪个更可能先发生？直接给出句子的数字序号，不要输出多余内容。"
    )

    response = glm4(prompt)

    if response == 2 or response == "2":
        return sentence+context
    else:
        return context+sentence

def process_list_num(input_list):
    processed_list = []
    for item in input_list:
        if isinstance(item, str) and not item.isdigit():
            first_number = re.search(r'\d+', item)
            if first_number:
                processed_list.append(float(first_number.group())) 
        elif isinstance(item, (int, float)) or item.isdigit():
            processed_list.append(float(item))  
    return processed_list


def extract_n(s):
    match = re.search(r'\d+(?:\.\d+)?', s)
    if match:
        number_str = match.group()
        return float(number_str) if '.' in number_str else int(number_str)
    else:
        return -1