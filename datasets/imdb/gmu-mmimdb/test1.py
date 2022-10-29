from transformers import pipeline
from transformers import AlbertTokenizer, AlbertModel

text = "Replace me by any text you'd like." * 100

print(len(text))
generator = pipeline(task="feature-extraction", model="albert-base-v2", truncation=True)
res = generator(text)
print(len(res), len(res[0]), len(res[0][0]))

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained("albert-base-v2")
encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
output = model(**encoded_input)
print(output.last_hidden_state[:,-1,:].size())