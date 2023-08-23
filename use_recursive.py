import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 从下载好的文件夹中加载tokenizer
# 这里你需要改为自己的实际文件夹路径
tokenizer = GPT2Tokenizer.from_pretrained('/luzm_ssd/codefornotes/gpt2_long_sentences_to_generate')
input_content = input('Please type in some words to begin\n')
predicted_text = input_content

# 从下载好的文件夹中加载预训练模型
model = GPT2LMHeadModel.from_pretrained('/luzm_ssd/codefornotes/gpt2_long_sentences_to_generate')

# 设置为evaluation模式，去取消激活dropout等模块。
model.eval()
max_len = 1000
if len(predicted_text) >= max_len:
    print(f'Reached the maximum length: {max_len}')

while len(predicted_text) <= max_len:
    # 预测所有token
    with torch.no_grad():
        # 编码一段文本
        indexed_tokens = tokenizer.encode(predicted_text)
        # 转换为pytorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])
        outputs = model(tokens_tensor)
        predictions = outputs[0]
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        print(predicted_text)

print(f'Reached the maximum length: {max_len}')

