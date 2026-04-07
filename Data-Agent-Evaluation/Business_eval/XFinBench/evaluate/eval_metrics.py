import os

def get_calcu_error_bool(corr_ans, model_ans):
    try:
        model_ans = float(model_ans)
        if model_ans==0:
            return 0
    except Exception as e:
        print(f"Error in get_calcu_error_bool() when tranferring model_ans to float() type! {e}")
        return 0
    try:
        corr_ans = float(corr_ans)
    except Exception as e:
        print(f"Error in get_calcu_error_bool() when tranferring corr_ans to float() type! {e}")
        return 0
    if abs((model_ans-corr_ans)/model_ans) < 0.005:
        return 1
    elif abs((model_ans*0.01-corr_ans)/(model_ans*0.01)) < 0.005 or abs((model_ans-corr_ans*0.01)/(model_ans)) < 0.005:
        return 1
    else:
        return 0

def get_calcu_bool(corr_ans, model_ans):
    try:
        model_ans = float(model_ans)
    except Exception as e:
        print(f"Error in get_calcu_bool() when tranferring model_ans to float() type! {e}")
        return 0
    try:
        corr_ans = float(corr_ans)
    except Exception as e:
        print(f"Error in get_calcu_bool() when tranferring corr_ans to float() type! {e}")
        return 0
    if model_ans==corr_ans:
        return 1
    else:
        return 0

def resp2ans(task, resp):
    if type(resp)!=str:
        return ''
    end_sent = 'Therefore, my answer is'
    if end_sent not in resp:
        return ''
    if  task=='bool':
        resp = resp.split(end_sent)[-1]
        resp_lower = resp.lower()
        if 'true' in resp_lower or 'correct' in resp_lower or '1' in resp_lower:
            return 1
        if 'false' in resp_lower or 'incorrect' in resp_lower or '0' in resp_lower:
            return 0
        return ''
    elif task=='mcq':
        resp = resp.split(end_sent)[-1].split('.')[0]
        if 'A' in resp:
            return 'A'
        elif 'B' in resp:
            return 'B'
        elif 'C' in resp:
            return 'C'
        elif 'D' in resp:
            return 'D'
        else:
            return ''
    elif task=='calcu':
        resp = resp.split(end_sent)[-1].strip()
        ans_ = resp.split('[')[-1].split(']')[0].strip()
        ans_1 = ''
        for chr_ in list(ans_):
            if ord(chr_) > 57 or ord(chr_) < 45:
                continue
            ans_1 += chr_
        try:
            ans_2 = float(ans_1)
            return ans_2
        except:
            return ''
