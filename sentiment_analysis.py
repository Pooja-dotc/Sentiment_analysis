import whisper
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random
import numpy as np

# Load the ASR model using Whisper
model = whisper.load_model("base")
options = whisper.DecodingOptions(fp16=False)
result = model.transcribe("C:\\Users\\kriti\\Downloads\\normal.aac")
print(result["text"])

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

text = result["text"]

tokens = tokenizer.encode_plus(
    text,
    max_length=128,
    truncation=True,
    padding='max_length',
    return_tensors='pt'
)
input_ids = tokens['input_ids'].to(device)
attention_mask = tokens['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# Apply softmax to the logits
probabilities = torch.softmax(logits, dim=1)

# Map class indices to sentiment labels
class_to_sentiment = {
    0: "positive",
    1: "neutral",
    2: "negative"
}

sentiment_label = torch.argmax(probabilities, dim=1).item()
sentiment = class_to_sentiment[sentiment_label]

print(f"Text: {text}")
print(f"Sentiment: {sentiment}")