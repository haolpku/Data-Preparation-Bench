# Colab - Sample_Experiment.ipynb - Data Quality Checking - Bool - Deduplicate
import os
import csv
import random
import numpy as np
import pandas as pd
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = "your_openai_api_key"
os.environ['PROJECT_PATH'] = "your_project_path"

project_path = os.environ["PROJECT_PATH"]
model_engine = 'gpt-4o-2024-05-13'
qa_data_bool_1 = pd.read_csv(project_path + 'QA_generation_then_verify_bool_true.csv')
qa_data_bool_0 = pd.read_csv(project_path + 'QA_generation_then_verify_bool_false.csv')
save_dir_1 = f'{project_path}/construct_data/QA_generation_then_verify_bool_true_dedup.csv'
save_dir_0 = f'{project_path}/construct_data/QA_generation_then_verify_bool_false_dedup.csv'
qa_data_bool_1['drop'] = 0
qa_data_bool_0['drop'] = 0
print(f"SAVE PATH:\n{save_dir_1}\n{save_dir_0}...")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
sys_msg = """
You are an expert in verifying question-answer pairs.
"""
with open(file=project_path + f'prompts/verify/bool_deduplicate.txt', mode='r', encoding='UTF-8') as fp:
    prompt_dedup = fp.read()

def call_chatgpt(msg, sys_msg, model_engine):
    completion = client.chat.completions.create(
        model= model_engine,
        messages=[
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": msg}
        ]
    )
    reply = completion.choices[0].message.content
    num_token = str(completion.usage.completion_tokens) + ';' + str(completion.usage.prompt_tokens)
    return reply, num_token

uni_id_list = qa_data_bool_1['uni_id'].unique().tolist()
for id_ in uni_id_list:
    print(f"\n\n## {id_}")
    
    bool_0_idx_list = qa_data_bool_0[(qa_data_bool_0['uni_id']==id_)&(qa_data_bool_0['verify_result']==1)].index.tolist()
    for idx_0 in bool_0_idx_list:
        bool_0_stat = qa_data_bool_0.loc[idx_0]['question']

        bool_1_idx_list = qa_data_bool_1[(qa_data_bool_1['uni_id']==id_)&(qa_data_bool_1['drop']==0)&(qa_data_bool_1['verify_result']==1)].index.tolist()
        for idx_1 in bool_1_idx_list:
            context_1 = qa_data_bool_1.loc[idx_1]['context']
            bool_1_stat = qa_data_bool_1.loc[idx_1]['question']
            prompt_id_dedup = prompt_dedup.format(context_1=context_1, statement_1=bool_1_stat, statement_0=bool_0_stat)
            reply, num_token = call_chatgpt(prompt_id_dedup, sys_msg, model_engine)
            print(f"\n{reply}")
            # Yes -> drop one, No -> keep both
            if 'no' in reply or 'No' in reply:
                continue

            num_bool_1 = len(bool_1_idx_list)
            num_bool_0 = len(qa_data_bool_0[(qa_data_bool_0['uni_id']==id_)&(qa_data_bool_0['drop']==0)&(qa_data_bool_0['verify_result']==1)])
            if num_bool_1==num_bool_0:
                # If # of true = # of false, drop one randomly
                if random.random() < 0.5:
                    qa_data_bool_0.loc[idx_0, 'drop'] = 1
                    print(f"[Equal] Drop idx_0 = {idx_0}")
                else:
                    qa_data_bool_1.loc[idx_1, 'drop'] = 1
                    print(f"[Equal] Drop idx_1 = {idx_1}")
            elif num_bool_1 > num_bool_0:
                # If # of true > # of false, drop true and lable it as drop=1
                qa_data_bool_1.loc[idx_1, 'drop'] = 1
                print(f"Drop idx_1 = {idx_1}")
            elif num_bool_1 < num_bool_0:
                # If # of true < # of false, drop false and lable it as drop=1
                qa_data_bool_0.loc[idx_0, 'drop'] = 1
                print(f"Drop idx_0 = {idx_0}")

qa_data_bool_1.to_csv(save_dir_1)
qa_data_bool_0.to_csv(save_dir_0)

print('===================================')
print(f"# of true statements: {len(qa_data_bool_1)}")
print(f"# of dropped true statements: {len(qa_data_bool_1[qa_data_bool_1['drop']==1])}")
print(f"# of left true statements: {len(qa_data_bool_1[(qa_data_bool_1['drop']==0)&(qa_data_bool_1['verify_result']==1)])}")
print('-----------------------------------')
print(f"# of false statements: {len(qa_data_bool_0)}")
print(f"# of dropped false statements: {len(qa_data_bool_0[qa_data_bool_0['drop']==1])}")
print(f"# of left false statements: {len(qa_data_bool_0[(qa_data_bool_0['drop']==0)&(qa_data_bool_0['verify_result']==1)])}")
print('===================================')