## Goal
This is a module to make data analysis and code development in generative ML experimentation more accessible and useful.

## Installation
```
git clone
```

## Examples
### Importing the module
```
# Import panml
from panml import ModelPack

# Import other required libraries
import numpy as np
import pandas as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
```

### Using HuggingFace models
Load model and tokenizer from HuggingFace
```
### Fetch GPT2 model from HuggingFace hub
tokenizer = AutoTokenizer.from_pretrained('distilgpt2', mirror='https://huggingface.co')
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```

### Generate output
```
output = model.predict('hello world is')
print(output['text'])
```
```
'hello world is a place where people can live and work together, and where people can live and work together, and where people can live and work together'
```

### Show probability of output token
```
output = model.predict('hello world is', display_probability=True)
print(output['probability'][:5]) # show probability of first 5 tokens in the generated output that follows the provided context
```
```
[{'token': ' a', 'probability': 0.052747420966625214},
 {'token': ' place', 'probability': 0.045980263501405716},
 {'token': ' where', 'probability': 0.4814596474170685},
 {'token': ' people', 'probability': 0.27657589316368103},
 {'token': ' can', 'probability': 0.2809840738773346}]
```
 
### Fine tune the model with your own data (in dataframe), training in self-supervised autoregressive format
```
# Specify train args
train_args = {
    'title': 'my_tuned_gpt2',
    'num_train_epochs' : 50,
    'mlm': False,
    'optimizer': 'adamw_torch',
    'per_device_train_batch_size': 10,
    'per_device_eval_batch_size': 10,
    'warmup_steps': 20,
    'weight_decay': 0.01,
    'logging_steps': 10,
    'output_dir': './results',
    'logging_dir': './logs',
    'save_model': True,
}

# Prepare data
x = df['some_text']
y = x

# Train model
model.fit(x, y, train_args, instruct=False)
```

### Run prediction
```
output = model.predict('hello world is', display_probability=True)
```

