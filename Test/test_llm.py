import unittest
import torch
import os
import json
import sys
import logging
import datetime
from Source.model.model_config import LlamaConfig
from Source.model.llama import LlamaModel
from Source.engine.llm_engine import LLMEngine
from Source.utils.logging_utils import setup_logging

# Set up log directory
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)

# Create unique log filename using timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"test_llm_{timestamp}.log")

# Set up logging, output to both console and file
setup_logging(log_level="INFO", log_file=log_file)
logger = logging.getLogger(__name__)
logger.info(f"Logs will be output to: {log_file}")

class TestLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logger.info("Starting to set up test environment")
        
        # Specify model path
        cls.model_dir = "/cluster/home/tangyz/InferLite/Models/Llama-3.2-1B-Instruct"
        
        # Check if model directory exists
        if not os.path.exists(cls.model_dir):
            logger.error(f"Model directory does not exist: {cls.model_dir}")
            raise FileNotFoundError(f"Model directory does not exist: {cls.model_dir}")
        
        # Load Hugging Face configuration
        hf_config_path = os.path.join(cls.model_dir, "config.json")
        if not os.path.exists(hf_config_path):
            logger.error(f"Configuration file does not exist: {hf_config_path}")
            raise FileNotFoundError(f"Configuration file does not exist: {hf_config_path}")
        
        logger.info(f"Loading configuration file: {hf_config_path}")
        # Convert to our configuration format
        cls.config = LlamaConfig.from_hf_config(hf_config_path)
        logger.info(f"Configuration loaded successfully")
        
        # Print key configuration parameters
        logger.info(f"Model type: {cls.config.model_type}")
        logger.info(f"Position encoding parameters: rope_theta={cls.config.rope_theta}")
        logger.info(f"Special tokens: BOS={cls.config.bos_token_id}, EOS={cls.config.eos_token_id}, PAD={cls.config.pad_token_id}")
        logger.info(f"Attention parameters: bias={cls.config.attention_bias}")
        logger.info(f"MLP parameters: bias={cls.config.mlp_bias}, act={cls.config.hidden_act}")
        
        # Create temporary directory
        os.makedirs("temp", exist_ok=True)
        
        # Save our configuration format
        cls.config_path = "temp/our_config.json"
        cls.config.to_json(cls.config_path)
        logger.info(f"Configuration saved to: {cls.config_path}")
        
        # Model weights path
        cls.model_path = os.path.join(cls.model_dir, "pytorch_model.bin")
        if not os.path.exists(cls.model_path):
            # Check if it's a safetensors model
            cls.model_path = os.path.join(cls.model_dir, "model.safetensors")
            if not os.path.exists(cls.model_path):
                # Check if there are sharded models
                shard_files = [f for f in os.listdir(cls.model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")]
                if shard_files:
                    logger.info(f"Detected sharded model: {shard_files}")
                    # Use the first shard for testing
                    cls.model_path = os.path.join(cls.model_dir, shard_files[0])
                    logger.info(f"Using first shard: {cls.model_path}")
                else:
                    logger.error(f"Model weights file does not exist: {cls.model_path}")
                    raise FileNotFoundError(f"Model weights file does not exist: {cls.model_path}")
        
        logger.info(f"Model weights path: {cls.model_path}")
        
        # Use model directory as tokenizer path
        cls.tokenizer_path = cls.model_dir
        logger.info(f"Tokenizer path: {cls.tokenizer_path}")
        logger.info("Test environment setup completed")
    
    def test_model_loading(self):
        """Test model loading"""
        logger.info("Starting to test model loading")
        try:
            # Create model
            model = LlamaModel(self.config)
            logger.info(f"Successfully created model: {model.__class__.__name__}")
            
            # Print model structure information
            logger.info(f"Model configuration: layers={self.config.num_layers}, hidden_size={self.config.hidden_size}")
            
            # Validate model structure
            self.assertEqual(len(model.layers), self.config.num_layers)
            
            # Test simple forward pass
            input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
            attention_mask = torch.ones_like(input_ids)
            # Ensure model is in inference mode
            model.eval()
            logger.info(f"Model set to inference mode: model.training={model.training}")
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            
            logger.info(f"Model forward pass successful, output shape: {outputs.shape}")
            self.assertEqual(outputs.shape, (1, 10, self.config.vocab_size))
            logger.info("Model loading test passed")
            
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Model loading test failed: {e}")
    
    def test_text_generation(self):
        """Test text generation"""
        logger.info("Starting to test text generation")
        try:
            # Create inference engine
            logger.info("Creating inference engine...")
            engine = LLMEngine(
                model_path=self.model_path,
                config_path="temp/our_config.json",
                tokenizer_path=self.tokenizer_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"Engine creation completed, model inference mode: training={engine.model.training}")
            
            # Import samplers
            from Source.sampler import SamplerFactory
            
            # Test different generation parameters
            prompt = "你好，请介绍一下自己。"
            
            # Test greedy decoding
            logger.info(f"\nTesting greedy decoding:")
            logger.info(f"Input prompt: {prompt}")
            greedy_sampler = SamplerFactory.create_greedy_sampler()
            greedy_response = engine.generate(
                prompt=prompt,
                sampler=greedy_sampler,
                max_length=50
            )
            logger.info(f"Greedy decoding result: {greedy_response}")
            
            # Test temperature sampling
            logger.info(f"\nTesting temperature sampling (temperature=0.7):")
            temp_sampler = SamplerFactory.create_temperature_sampler(temperature=0.7)
            temp_response = engine.generate(
                prompt=prompt,
                sampler=temp_sampler,
                max_length=50
            )
            logger.info(f"Temperature sampling result: {temp_response}")
            
            # Test top-p sampling
            logger.info(f"\nTesting top-p sampling (top_p=0.9):")
            top_p_sampler = SamplerFactory.create_nucleus_sampler(p=0.9)
            top_p_response = engine.generate(
                prompt=prompt,
                sampler=top_p_sampler,
                max_length=50
            )
            logger.info(f"top-p sampling result: {top_p_response}")
            
            # Test top-k sampling
            logger.info(f"\nTesting top-k sampling (top_k=50):")
            top_k_sampler = SamplerFactory.create_top_k_sampler(k=50)
            top_k_response = engine.generate(
                prompt=prompt,
                sampler=top_k_sampler,
                max_length=50
            )
            logger.info(f"top-k sampling result: {top_k_response}")
            
            # Test repetition penalty
            logger.info(f"\nTesting repetition penalty (repetition_penalty=1.2):")
            base_sampler = SamplerFactory.create_combined_sampler(temperature=0.7, top_p=0.9)
            rep_penalty_sampler = SamplerFactory.add_repetition_penalty(base_sampler, penalty=1.2)
            rep_penalty_response = engine.generate(
                prompt=prompt,
                sampler=rep_penalty_sampler,
                max_length=50
            )
            logger.info(f"Repetition penalty result: {rep_penalty_response}")
            
            # Test English prompt
            prompt = "What is the capital of France?"
            logger.info(f"\nTesting English prompt:")
            logger.info(f"Input prompt: {prompt}")
            
            # Create sampler using factory config method
            sampler_config = {
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True
            }
            default_sampler = SamplerFactory.create_sampler(sampler_config)
            
            response = engine.generate(
                prompt=prompt,
                sampler=default_sampler,
                max_length=50
            )
            
            logger.info(f"Model response: {response}")
            
            # Verify all outputs are not empty and of correct type
            for resp in [greedy_response, temp_response, top_p_response, top_k_response, rep_penalty_response, response]:
                self.assertIsInstance(resp, str)
                self.assertTrue(len(resp) > 0)
            
            logger.info("Text generation test passed")
            
        except Exception as e:
            logger.error(f"Text generation test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Text generation test failed: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        logger.info("Starting to clean up test environment")
        # Delete temporary files
        if os.path.exists(cls.config_path):
            os.remove(cls.config_path)
            logger.info(f"Deleted temporary configuration file: {cls.config_path}")
        if os.path.exists("temp"):
            os.rmdir("temp")
            logger.info("Deleted temporary directory")
        logger.info("Test environment cleanup completed")

if __name__ == "__main__":
    unittest.main() 