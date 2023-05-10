import unittest
from panml.models import ModelPack

class TestModelPack(unittest.TestCase):
    '''
    Run tests on ModelPack class in models.py
    '''
    def test_invalid_source_input(self):
        # test invalid source as int
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt2', source=1)
        # test invalid source as float
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt2', source=0.2)
        # test invalid source as non accepted str
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt2', source='huggingface1')

    def test_invalid_model_input(self):
        # test invalid model as int
        with self.assertRaises(ValueError):
            m = ModelPack(model=1, source='huggingface')
        # test invalid model as float
        with self.assertRaises(ValueError):
            m = ModelPack(model=0.2, source='huggingface')
        # test invalid model as non accepted str
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt99', source='huggingface')

    def test_invalid_model_source_input(self):
        # test invalid model and source match combo 1
        with self.assertRaises(ValueError):
            m = ModelPack(model='text-davinci-002', source='huggingface')
        # test invalid model and source match combo 2
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt2', source='openai')

    def test_correct_model_source_input(self):
        # test valid model and source match combo 1
        m = ModelPack(model='gpt2', source='huggingface')

        # test valid model and source match combo 2
        m = ModelPack(model='tezt-davinci-002', source='openai')
        
