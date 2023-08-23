import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 从下载好的文件夹中加载tokenizer
# 这里你需要改为自己的实际文件夹路径
tokenizer = GPT2Tokenizer.from_pretrained('/luzm_ssd/codefornotes/gpt2_long_sentences_to_generate')
input_content = input()
# text = 'Who was Jim Henson ? Jim Henson was a'
text = input_content
# 编码一段文本
# 编码后为[8241, 373, 5395, 367, 19069, 5633, 5395, 367, 19069, 373, 257]
indexed_tokens = tokenizer.encode(text)
# 转换为pytorch tensor
# tensor([[ 8241,   373,  5395,   367, 19069,  5633,  5395,   367, 19069,   373, 257]])
# shape为 torch.Size([1, 11])
tokens_tensor = torch.tensor([indexed_tokens])
# 从下载好的文件夹中加载预训练模型
model = GPT2LMHeadModel.from_pretrained('/luzm_ssd/codefornotes/gpt2_long_sentences_to_generate')

# 设置为evaluation模式，去取消激活dropout等模块。
# 在huggingface/transformers框架中，默认就是eval模式
model.eval()

# 预测所有token
with torch.no_grad():
    # 将输入tensor输入，就得到了模型的输出，非常简单
    # outputs是一个元组，所有huggingface/transformers模型的输出都是元组
    # 本初的元组有两个，第一个是预测得分（没经过softmax之前的，也叫作logits），
    # 第二个是past，里面的attention计算的key value值
    # 此时我们需要的是第一个值
    outputs = model(tokens_tensor)
    # predictions shape为 torch.Size([1, 11, 50257])，
    # 也就是11个词每个词的预测得分（没经过softmax之前的）
    # 也叫做logits
    predictions = outputs[0]

# 我们需要预测下一个单词，所以是使用predictions第一个batch，最后一个词的logits去计算
# predicted_index = 582，通过计算最大得分的索引得到的
predicted_index = torch.argmax(predictions[0, -1, :]).item()
# 反向解码为我们需要的文本
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
# 解码后的文本：'Who was Jim Henson? Jim Henson was a man'
# 成功预测出单词 'man'
print(predicted_text)