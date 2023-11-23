from transformers import BertTokenizer, BertModel
import torch
import time 

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def prediction(sentences, tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
# Load model from HuggingFace Hub

tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')

duration_list = []
for i in range(10):
    start = time.time()
    sentence_embeddings = prediction(sentences, tokenizer, model)
    end = time.time()
    duration = time.time() - start
    duration_list.append(duration)   
    print(sentence_embeddings)

print(f"total time: {duration_list}")

last_five = duration_list[5:]
print(f"average time: {sum(last_five)/len(last_five)}")