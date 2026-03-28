from llamafactory.chat import register_template

register_template(
    name="my_llama3",  
    system_prompt="You are a helpful assistant.",
    template=dict(
        prefix="<|start_header_id|>user<|end_header_id|>\n{instruction}",
        suffix="<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n{output}<|eot_id|>",
        separator="\n输入：{input}" if "{input}" in "{instruction}" else "",
        stop_words=["<|eot_id|>"]
    )
)
