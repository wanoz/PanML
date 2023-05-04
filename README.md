## PanML: A simplified generative AI/ML development toolkit

## Goal
This package aims to make analysis and experimentation of generative AI/ML models more accessible, by providing a consistent layer of programmatic interface to foundation models for Data Scientists, Machine Learning Engineers and Software Developers. It's a work in progress, so very much open for collaboration and contribution. 

## Installation
```
git clone https://github.com/wanoz/panml.git
```

## Usage
### Importing the module
```
# Import panml
from panml.models import ModelPack

# Import other modules/packages as required
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
...
```

### Using HuggingFace models
Load model and tokenizer from HuggingFace
```
# Fetch GPT2 model from HuggingFace hub
tokenizer = AutoTokenizer.from_pretrained('distilgpt2', mirror='https://huggingface.co')
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```

Create model pack from the loaded model and tokenizer
```
modelpack = ModelPack(model, tokenizer, input_block_size=20, source='huggingface')
```

Generate output
```
output = modelpack.predict('hello world is')
print(output['text'])
```
```
# Output
'hello world is a place where people can live and work together, and where people can live and work together, and where people can live and work together'
```

Show probability of output token
```
output = modelpack.predict('hello world is', display_probability=True)
print(output['probability'][:5]) # show probability of first 5 tokens in the generated output that follows the provided context
```
```
# Output
[{'token': ' a', 'probability': 0.052747420966625214},
 {'token': ' place', 'probability': 0.045980263501405716},
 {'token': ' where', 'probability': 0.4814596474170685},
 {'token': ' people', 'probability': 0.27657589316368103},
 {'token': ' can', 'probability': 0.2809840738773346}]
```
 
Fine tune the model with your own data from Pandas dataframe - execute in self-supervised autoregressive training regime.
```
# Specify train args
train_args = {
    'title': 'my_tuned_gpt2',
    'num_train_epochs' : 30,
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
modelpack.fit(x, y, train_args, instruct=False)
```

Generate output with the fine tuned model
```
output = modelpack.predict('hello world is', display_probability=True)
print(output['text'])
```

### Using OpenAI models
Create model pack from OpenAI model description and API key
```
modelpack = ModelPack(model='text-davinci-002', source='openai', api_key=<your_openai_key>)
```

Generate output
```
output = modelpack.predict('What is the best way to live a healthy lifestyle?')
output['text']
```
```
# Output
\nThe best way to live a healthy lifestyle is to eat healthy foods, get regular exercise, and get enough sleep.
```

Show probability of output token
```
output = modelpack.predict('What is the best way to live a healthy lifestyle?', display_probability=True)
print(output['probability'][:5]) # show probability of first 5 tokens in the generated output that follows the provided context
```
```
# Output
[{'token': '\n', 'probability': 0.9912449516093955},
 {'token': 'The', 'probability': 0.40432789860673046},
 {'token': ' best', 'probability': 0.9558591494467851},
 {'token': ' way', 'probability': 0.9988543268851316},
 {'token': ' to', 'probability': 0.9993104225678759}]
```

Generate output in prompt modified loop (using a prompt modifier)
```
prompt_modifier = [
    {"pre": "You are a mother", 
     "post": ""},
    {"pre": "breakdown into further details", 
     "post": ""},
    {"pre": "summarise to answer the original question", 
     "post": ""},
]

output = modelpack.predict('What is the best way to live a healthy lifestyle?', prompt_modifier=prompt_modifier)
print(output['text'])
```
```
# Output
'\n\nTo live a healthy lifestyle, individuals should eat a variety of healthy foods, exercise regularly, get enough sleep, and avoid tobacco products, excessive alcohol, and other drugs.'
```

Generate embedding
```
output = modelpack.embedding('What is the best way to live a healthy lifestyle?')
print(output[:5]) # show first 5 embedding elements
```
```
# Output
[0.025805970653891563,
 0.007422071415930986,
 0.01738160289824009,
 -0.006787706166505814,
 -0.003324073040857911]
```
