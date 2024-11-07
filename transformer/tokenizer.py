from transformers import BertModel, AutoTokenizer
import pandas as pd

model_name='bert-base-cased'

model = BertModel.from_pretrained(model_name)
