import os
import csv
import random
import numpy as np
import pandas as pd
from openai import OpenAI
from deep_translator import GoogleTranslator

os.environ['OPENAI_API_KEY'] = "your_openai_api_key"
os.environ['PROJECT_PATH'] = "your_project_path"

task_ = 'calcu' # bool_true, bool_false, mcq, calcu
model_engine =  'gpt-4o-2024-05-13'
project_path = os.environ["PROJECT_PATH"]
save_dir = f"{project_path}/construct_data/QA_generation_then_verify_{task_}.csv"
print(f"\n\nSAVE PATH: {project_path}{save_dir}...")

# Load dataset
qa_data = pd.read_csv(project_path + 'QA_for_Prompt.csv', encoding='utf-8')
qa_data['context'] = ''
qa_data['question'] = ''
qa_data['choice'] = ''
qa_data['answer'] = ''
qa_data['reply_gen'] = ''
qa_data['reply_verify'] = ''
qa_data['verify_result'] = 0 # 0 for dropping, and 1 for keeping
qa_data['reply_gen_token'] = '' # For calculating cost
qa_data['reply_verify_token'] = '' # For calculating cost
qa_type = ['bool', 'mcq', 'calcu',]
qa_type_index = []
for type_ in qa_type:
    qa_temp = qa_data['qa_type'].str.contains(type_).tolist()
    index_temp = [i for i,x in enumerate(qa_temp) if x == True]
    qa_type_index.append(index_temp)

print(f"# of bool QAs: {len(qa_type_index[0])}")
print(f"# of mcq QAs: {len(qa_type_index[1])}")
print(f"# of calcu QAs: {len(qa_type_index[2])}")

# Load prompts
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
sys_msg_gen = """
You are an expert in generating question-answer pairs.
"""
sys_msg_ver = """
You are an expert in verifying question-answer pairs.
"""
with open(file=head_dir + f'prompts/generate/{task_}.txt', mode='r', encoding='UTF-8') as fp:
    prompt_gen = fp.read()
with open(file=head_dir + f'prompts/verify/{task_}.txt', mode='r', encoding='UTF-8') as fp:
    prompt_verify = fp.read()

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
    
def reply_to_qa(task_, reply):
    if task_=='bool_true' or task_=='bool_false':
        sent_list = [item.replace('**', '').strip() for item in reply.split('Statement:') if len(item.strip())!=0]
        context_ = ''
        stat_list = []
        for sent in sent_list:
            if 'Context:' in sent:
                context_ = sent.replace('Context:', '').strip()
                continue
            sent_1 = sent.split('Answer:')[0].strip()
            stat_list.append(sent_1)
        return context_, stat_list
    elif task_=='mcq':
        sent_list = [item.replace('**', '').strip() for item in reply.split("Generated Question:") if len(item.strip())!=0]
        context_ = ''
        qa_list = []
        for sent in sent_list:
            if 'Context:' in sent:
                context_ = sent.replace('Context:', '').strip()
                continue
            question_ = sent.split('Choices:')[0].strip()
            if 'Correct Answer:' in sent:
                choices_ = sent.split('Choices:')[1].split('Correct Answer:')[0].strip()
                answer_ = sent.split('Correct Answer:')[1].strip()
            else:
                choices_ = sent.split('Choices:')[1].strip()
                answer_ = 'a'
            qa_list.append([question_, choices_, answer_])
        return context_, qa_list
    elif task_=='calcu':
        sent_list = [item.replace('**', '').strip() for item in reply.split("Generated Question:") if len(item.strip())!=0]
        context_ = ''
        qa_list = []
        for sent in sent_list:
            if 'Context:' in sent:
                context_ = sent.replace('Context:', '').strip()
                continue
            question_ = sent.split('Answer:')[0].strip()
            answer_ = sent.split('Answer:')[1].strip()
            qa_list.append([question_, answer_])            
        return context_, qa_list

def shuffle_choices(choi_):
    # This function relies on the fact that "All correct answer is located at (a)."
    choi_a = choi_.split('(b)')[0].replace('(a)','').replace('\n','').strip()
    choi_b = choi_.split('(c)')[0].split('(b)')[1].replace('(b)','').replace('\n','').strip()
    choi_c = choi_.split('(c)')[1].replace('(c)','').replace('\n','').strip()
    shuffle_choice = [choi_a, choi_b, choi_c]
    random.shuffle(shuffle_choice)
    corr_idx = shuffle_choice.index(choi_a)

    dict_choi = ['A', 'B', 'C']
    shuffle_choi_str = '\n'.join([dict_choi[k]+'. '+shuffle_choice[k] for k in range(len(dict_choi))])
    shuffle_ans = dict_choi[corr_idx]
    return shuffle_choi_str, shuffle_ans

def check_choices_length(choi_, limit_=5):
    choices_list = [item.strip() for item in choi_.split('\n') if len(item.strip())!=0]
    choices_len_list = [len(item.split(' ')) for item in choices_list]
    if max(choices_len_list)-min(choices_len_list)>limit_:
        return 0
    else:
        return 1

def determine_only_number(str_):
    flg = 1
    for ch_ in str_:
        if ord(ch_)>=45 and ord(ch_)<=57:
            continue
        else:
            flg = 0
            break
    return flg


# Process all of QAs
qa_idx = qa_type.index(task_.split('_')[0].strip())
sample_df = pd.DataFrame(columns=qa_data.columns)
if task_!='bool_false':
    sample_id_list = qa_type_index[qa_idx]
else:
    # Make sure the qa pairs in bool true and bool false are the same ones
    sample_tmp = pd.read_csv(save_dir.replace('false', 'true'))
    uni_id_list_tmp = sample_tmp['uni_id'].unique().tolist()
    sample_id_list = [qa_data[qa_data['uni_id']==item].index.tolist()[0] for item in uni_id_list_tmp]

# Loop each original question-answer pair
flag = 0
for sample_id in sample_id_list:
    print(f"\nProcessing INDEX= {sample_id}..\n")
    orig_row = qa_data.loc[sample_id]
    orig_ques = qa_data.loc[sample_id]['ques']
    orig_ans = qa_data.loc[sample_id]['ans']

    # Generate QA
    prompt_id_gen = prompt_gen.format(orig_ques=orig_ques, orig_ans=orig_ans)
    reply, num_token = call_chatgpt(prompt_id_gen, sys_msg_gen, model_engine)
    print(f'{reply}\n')
    context_, qa_list = reply_to_qa(task_, reply)
    if len(qa_list)==0:
        sample_df.loc[flag] = orig_row
        sample_df.loc[flag, 'reply_gen'] = reply
        sample_df.loc[flag, 'reply_gen_token'] = num_token
        flag += 1
        continue

    # Verify QA and Save them
    for qa_ in qa_list:
        if task_=='bool_true' or task_=='bool_false':
            prompt_id_verify = prompt_verify.format(orig_ques=orig_ques, orig_ans=orig_ans, context=context_, question=qa_)
            reply_verify, num_token_verify = call_chatgpt(prompt_id_verify, sys_msg_ver, model_engine)
            sample_df.loc[flag] = orig_row
            sample_df.loc[flag, 'context'] = context_
            sample_df.loc[flag, 'question'] = qa_
            sample_df.loc[flag, 'answer'] = 1 if task_=='bool_true' else 0
        elif task_=='mcq':
            prompt_id_verify = prompt_verify.format(orig_ques=orig_ques, orig_ans=orig_ans, context=context_, question=qa_[0], choices=qa_[1], answer=qa_[2])
            reply_verify, num_token_verify = call_chatgpt(prompt_id_verify, sys_msg_ver, model_engine)
            shuffle_choi, shuffle_ans = shuffle_choices(qa_[1])
            if check_choices_length(shuffle_choi)==0:
                reply_verify += '\nQ5: No, unqualified choice lengths.'
            sample_df.loc[flag] = orig_row
            sample_df.loc[flag, 'context'] = context_
            sample_df.loc[flag, 'question'] = qa_[0]
            sample_df.loc[flag, 'choice'] = shuffle_choi
            sample_df.loc[flag, 'answer'] = shuffle_ans
        elif task_=='calcu':
            prompt_id_verify = prompt_verify.format(orig_ques=orig_ques, orig_ans=orig_ans, context=context_, question=qa_[0], answer=qa_[1])
            reply_verify, num_token_verify = call_chatgpt(prompt_id_verify, sys_msg_ver, model_engine)
            if determine_only_number(qa_[1])==0:
                reply_verify += "\nQ3: No, unqualified final answer."
            sample_df.loc[flag] = orig_row
            sample_df.loc[flag, 'context'] = context_
            sample_df.loc[flag, 'question'] = qa_[0]
            sample_df.loc[flag, 'answer'] = qa_[1]

        sample_df.loc[flag, 'reply_gen'] = reply
        sample_df.loc[flag, 'reply_verify'] = reply_verify
        sample_df.loc[flag, 'verify_result'] = 0 if ('no' in reply_verify) or ('No' in reply_verify) else 1
        sample_df.loc[flag, 'reply_gen_token'] = num_token
        sample_df.loc[flag, 'reply_verify_token'] = num_token_verify
        flag += 1

sample_df = sample_df.rename(columns={'ques': 'orig_ques', 'ans': 'orig_ans', })
sample_df.to_csv(save_dir)