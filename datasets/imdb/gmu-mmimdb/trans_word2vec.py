from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import numpy as np
# Sentiment analysis pipeline
pipeline('sentiment-analysis')

# Question answering pipeline, specifying the checkpoint identifier
pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

# Named entity recognition pipeline, passing in a specific model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
res=pipeline('feature-extraction', model=model, tokenizer=tokenizer)

output = res('These compounds are present in etiolated tissue of seedlings grown in darkness.')
print(output)
output=np.array(output)
flat_output=output.ravel()
print(flat_output)
print(flat_output.shape)
print(flat_output.shape[0])
if flat_output.shape[0]<300:

    flat_output=np.concatenate((flat_output,np.zeros(300-flat_output.shape[0])))
else:
    flat_output=flat_output[:300]
print(flat_output)
print(flat_output.shape)
print(flat_output.shape[0])