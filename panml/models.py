import os
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import openai

# HuggingFace model
class HuggingFaceModelPack():
    '''
    Generic model pack class for HuggingFace Hub models
    '''
    # Initialize class variables
    def __init__(self, model, input_block_size, padding_length, tokenizer_batch, source):
        if source == 'huggingface':
            self.model_hf = AutoModelForCausalLM.from_pretrained(model)
        elif source == 'local':
            self.model_hf = AutoModelForCausalLM.from_pretrained(model, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf.config.model_type, mirror='https://huggingface.co')
        self.padding_length = padding_length
        self.input_block_size = input_block_size
        self.tokenizer_batch = tokenizer_batch
        self.train_default_args = ['title', 'num_train_epochs', 'optimizer', 'mlm', 
                                   'per_device_train_batch_size', 'per_device_eval_batch_size',
                                   'warmup_steps', 'weight_decay', 'logging_steps', 
                                   'output_dir', 'logging_dir', 'save_model']
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.model_hf.config.eos_token_id
    
    # Embed text
    def embedding(self, text):
        token_ids = self.tokenizer.encode(text, return_tensors='pt')
        
        # Get embeddings
        if 'flan' in self.model_hf.name_or_path: 
            emb = self.model_hf.shared.weight[token_ids[0]] # embeddings for FLAN
        elif ('gpt2' in self.model_hf.name_or_path) or ('Cerebras-GPT' in self.model_hf.name_or_path): 
            emb = self.model_hf.transformer.wte.weight[token_ids[0]] # embeddings for GPT2
            
        emb /= emb.norm(dim=1).unsqueeze(1) # normalise embedding weights
        emb_pad = torch.zeros(self.padding_length, emb.shape[1]) # Set and apply padding to embeddings
        emb_pad[:emb.shape[0], :] = emb
        
        return emb_pad
    
    # Generate text
    def predict(self, text, max_length=50, skip_special_tokens=True, display_probability=False, 
                num_return_sequences=1, temperature=0.8, top_p=0.8, top_k=0, num_beams=3, no_repeat_ngram_size=3, early_stopping=True):
        output_context = {
            'text': None,
            'probability': None,
        }
        
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        output = self.model_hf.generate(input_ids, 
                                        max_length=max_length,
                                        pad_token_id=self.model_hf.config.eos_token_id,
                                        num_return_sequences=num_return_sequences, 
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                        num_beams=num_beams,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        early_stopping=early_stopping,
                                        output_scores=display_probability, 
                                        return_dict_in_generate=display_probability, 
                                        renormalize_logits=True)
                                          
        # Get probability of output tokens
        if display_probability:
            output_context['text'] = self.tokenizer.decode(output['sequences'][0], skip_special_tokens=skip_special_tokens)
            output_context['probability'] = [
                {'token': self.tokenizer.decode(torch.argmax(math.e**(s)).item()), 
                 'probability': torch.max(math.e**(s)).item()} for s in output['scores']
            ]
        else:
            output_context['text'] = self.tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
        output_context['text'] = output_context['text'].replace('\n', '')
        output_context['text'] = output_context['text'].strip()
        
        return output_context
    
    # Tokenize function
    def _tokenize_function(self, examples):
        return self.tokenizer(examples['text'])
    
    # Tokenize pandas dataframe feature
    def tokenize_text(self, x, batched):  
        df_sample = pd.DataFrame({'text': x})
        hf_dataset = Dataset.from_pandas(df_sample)
        if batched:
            tokenized_dataset = hf_dataset.map(self._tokenize_function, batched=batched, num_proc=4)
        else:
            tokenized_dataset = hf_dataset.map(self._tokenize_function)

        return tokenized_dataset
    
    # Model training
    def fit(self, x, y, train_args={}, instruct=False):
        # Convert to tokens format from pandas dataframes
        tokenized_data = self.tokenize_text(x, batched=self.tokenizer_batch)
        tokenized_target = self.tokenize_text(y, batched=self.tokenizer_batch)
        
        # Check for missing input arguments
        assert set(list(train_args.keys())) == set(self.train_default_args), \
            f'Train args are not in the required format - missing: {", ".join(list(set(self.train_default_args) - set(list(train_args.keys()))))}'
        
        if instruct:
            print('Setting up training in sequence to sequence format...')
            tokenized_data = tokenized_data.add_column('labels', tokenized_target['input_ids']) # Create target sequence labels
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model) # Organise data for training
            
            # Setup training in sequence to sequence format
            training_args = TrainingArguments(
                optim=train_args['optimizer'], # model optimisation function
                num_train_epochs=train_args['num_train_epochs'], # total number of training epochs
                per_device_train_batch_size=train_args['per_device_train_batch_size'],  # batch size per device during training
                per_device_eval_batch_size=train_args['per_device_eval_batch_size'],   # batch size for evaluation
                warmup_steps=train_args['warmup_steps'], # number of warmup steps for learning rate scheduler
                weight_decay=train_args['weight_decay'], # strength of weight decay
                logging_steps=train_args['logging_steps'],
                output_dir='./results', # output directory
                logging_dir=train_args['logging_dir'], # log directory
            )

            trainer = Trainer(
                model=self.model_hf,
                args=training_args,
                train_dataset=tokenized_data.remove_columns(['text']),
                eval_dataset=tokenized_data.remove_columns(['text']),
                data_collator=data_collator,
            )
    
        else:
            print('Setting up training in autoregressive format...')
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=train_args['mlm']) # Organise data for training
            
            # Setup training in autoregressive format
            training_args = TrainingArguments(
                optim=train_args['optimizer'], 
                num_train_epochs=train_args['num_train_epochs'], 
                per_device_train_batch_size=train_args['per_device_train_batch_size'],
                per_device_eval_batch_size=train_args['per_device_eval_batch_size'], 
                warmup_steps=train_args['warmup_steps'], 
                weight_decay=train_args['weight_decay'], 
                logging_steps=train_args['logging_steps'],
                output_dir=train_args['output_dir'],
                logging_dir=train_args['logging_dir'], 
            )
            
            trainer = Seq2SeqTrainer(
                model=self.model_hf,
                args=training_args,
                train_dataset=tokenized_data.remove_columns(['text']),
                eval_dataset=tokenized_data.remove_columns(['text']),
                data_collator=data_collator,
            )

        trainer.train() # Execute training
        
        if train_args['save_model']:
            trainer.save_model(f'./results/model_{train_args["title"]}') # Save trained model
    
# OpenAI model
class OpenAIModelPack():
    '''
    OpenAI model class
    '''
    def __init__(self, model, api_key):
        self.model = model
        self.model_embedding = 'text-embedding-ada-002'
        openai.key = api_key
    
    # Generate text of single model call
    @staticmethod
    def _predict(model, text, temperature, max_tokens, top_p, n, frequency_penalty, presence_penalty, 
                 display_probability, logprobs):
        output_context = {
            'text': None,
            'probability': None,
        }
        response = openai.Completion.create(
            model=model,
            prompt=text,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs
        )

        output_context['text'] = response['choices'][0]['text']

        # Get probability of output tokens
        if display_probability:
            tokens = response["choices"][0]['logprobs']['tokens']
            token_logprobs = response["choices"][0]['logprobs']['token_logprobs']
            output_context['probability'] = [{'token': token, 'probability': math.e**logprob} for token, logprob in zip(tokens, token_logprobs)]

        return output_context
    
    # Generate text in prompt loop
    def predict(self, text, temperature=0, max_tokens=100, top_p=1, n=3, frequency_penalty=0, presence_penalty=0, 
                display_probability=False, logprobs=1, prompt_modifier=[{'pre': '', 'post': ''}], keep_last=True):
        
        # Create loop for text prediction
        response_words = 0
        history = []
        for count, mod in enumerate(prompt_modifier):
            if count > 0:
                text = output_context['text']
            text = f"{mod['pre']} \n {text} \n {mod['post']}"
            output_context = self._predict(self.model, text, temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                                           n=n, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                                           display_probability=display_probability, logprobs=logprobs)

            # Terminate loop for next prompt when context contains no meaningful words (less than 2)
            response_words = output_context['text'].replace('\n', '').replace(' ', '')
            if len(response_words) < 2:
                break

            history.append(output_context)
        
        if keep_last:
            return history[-1] # returns last prediction output
        else:
            return history # returns all historical prediction output
        
    # Embed text
    def embedding(self, text, model=None):
        text = text.replace("\n", " ")
        if model is None:
            model = self.model_embedding          
        emb = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
        
        return emb
        
# Entry model pack class           
class ModelPack():
    '''
    Main model pack class
    '''
    def __init__(self, 
                 model,
                 tokenizer=None, 
                 input_block_size=10, 
                 padding_length=100, 
                 tokenizer_batch=False,
                 source='huggingface', 
                 api_key=None):
        
        self.padding_length = padding_length
        self.tokenizer = tokenizer
        self.model = model
        self.input_block_size = input_block_size
        self.tokenizer_batch = tokenizer_batch
        self.source = source
        self.api_key = api_key
        
        # Accepted models from sources
        self.accepted_models = {
            'huggingface': [
                'gpt2',
                'gpt2-medium',
                'gpt2-xl',
                'distilgpt2', 
                'google/flan-t5-base',
                'google/flan-t5-small',
                'google/flan-t5-large',
                'google/flan-t5-xl',
                'google/flan-t5-xxl',
                'cerebras/Cerebras-GPT-111M',
                'cerebras/Cerebras-GPT-256M',
                'cerebras/Cerebras-GPT-590M',
                'cerebras/Cerebras-GPT-1.3B',
                'cerebras/Cerebras-GPT-2.7B',
                'cerebras/Cerebras-GPT-6.7B',
                'cerebras/Cerebras-GPT-13B',
            ],
            'openai': [
                'text-davinci-002', 
                'text-davinci-003',
            ],
        }

        # Accepted source descriptions
        self.accepted_sources = [
            'huggingface', 
            'local', 
            'openai',
        ]
        
        # HuggingFace Hub model call
        assert self.source in self.accepted_sources, \
            'The specified source is not recognized. Supported sources are: ' + ' '.join([f"{s}" for s in self.accepted_sources])
        if self.source == 'huggingface':
            assert self.model in self.accepted_models['huggingface'], \
                'The specified model is currently not supported in this package. Supported HuggingFace Hub models are: ' + ' '.join([f"{m}" for m in self.accepted_models['huggingface']])
            self.instance = HuggingFaceModelPack(self.model, self.input_block_size, self.padding_length, self.tokenizer_batch, self.source)

        # Locally trained model call of HuggingFace Hub model
        elif self.source == 'local':
            self.instance = HuggingFaceModelPack(self.model, self.input_block_size, self.padding_length, self.tokenizer_batch, self.source)

        # OpenAI model call
        elif self.source == 'openai':
            assert self.model in self.accepted_models['openai'], \
                'The specified model currently is not supported in this pacckage. Supported OpenAI models are: ' + ' '.join([f"{m}" for m in self.accepted_models['openai']])
            assert self.api_key is not None, 'api key has not been specified for OpenAI model call'
            self.instance = OpenAIModelPack(model=self.model, api_key=self.api_key)

    # Direct to the attribute ofthe sub model pack class (attribute not found in the main model pack class)
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
    
    
# Set device type
def set_device():
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_type

# Calculate cosine similarity of two matrices    
def cosine_similarity(m1, m2):
    return F.cosine_similarity(m1.view(1, -1), m2.view(1, -1))

