import os
import base64
from typing import List, Dict

from openai import OpenAI


# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://XXX/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                            
REFER_API_BASE_URL = "http://XXX/v1"                  
REFER_API_KEY = "sk-dummy"  
# ========================================================


class build_model:
    def __init__(self, model_name: str,):
        self.project_path = os.environ.get("PROJECT_PATH", ".")
        self.client = OpenAI(
            api_key=NEW_API_KEY,
            base_url=NEW_BASE_URL
        )

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _build_messages(self, sys_msg: str, msg: str, image_pt: str, sys_msg_bool: int) -> List[Dict]:
        messages: List[Dict] = []
        if sys_msg_bool == 1 and sys_msg:
            messages.append({"role": "system", "content": sys_msg})

        user_content: List[Dict] = []
        if image_pt:
            base64_image = self.encode_image(image_pt)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        user_content.append({"type": "text", "text": msg})
        messages.append({"role": "user", "content": user_content})
        return messages

    def get_model_response(self, sys_msg, msg, model_name, image_pt='', sys_msg_bool=1, max_token_=1024):
        messages = self._build_messages(sys_msg, msg, image_pt, sys_msg_bool)
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_token_,
        )
        reply = completion.choices[0].message.content
        num_token = f"{completion.usage.completion_tokens};{completion.usage.prompt_tokens}"
        return reply, num_token
