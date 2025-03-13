import unittest
import torch
import os
import sys
import logging
import datetime
import time

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Source.utils import setup_logging
from Source.mem_manage import MemoryMonitor
from Source.engine import LLMEngine
from Source.sampler import SamplerFactory
from Source.model.model_config import LlamaConfig
from Source.model.llama import LlamaModel

# Set up log directory
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)

# Create unique log filename using timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"test_memory_monitor_{timestamp}.log")

# Set up logging, output to both console and file
setup_logging(log_level="INFO", log_file=log_file)
logger = logging.getLogger(__name__)
logger.info(f"Logs will be output to: {log_file}")

class TestMemoryMonitor(unittest.TestCase):
    """Test memory monitoring functionality"""
    
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
        
        # Check if GPU is available
        cls.has_cuda = torch.cuda.is_available()
        if not cls.has_cuda:
            logger.warning("No CUDA device available, skipping GPU memory monitoring tests")
        
        logger.info("Test environment setup completed")
    
    def test_memory_monitor_basic(self):
        """Test basic memory monitoring functionality"""
        if not self.has_cuda:
            logger.info("Skipping test: No CUDA device available")
            return
            
        # Create memory monitor
        monitor = MemoryMonitor(enabled=True, interval=2)
        monitor.start()
        
        # Allocate some GPU memory
        tensors = []
        for i in range(5):
            # Allocate about 100MB each time
            tensor = torch.zeros(25 * 1024 * 1024, device='cuda')  # About 100MB
            tensors.append(tensor)
            monitor.step()
            time.sleep(0.1)  # Short delay
        
        # Release some memory
        tensors = tensors[:2]  # Keep only the first two tensors
        torch.cuda.empty_cache()  # Clear cache
        monitor.step(force_log=True)
        
        # Stop monitoring
        stats = monitor.stop()
        
        # Validate results
        self.assertIsNotNone(stats)
        self.assertTrue('peak_memory' in stats)
        self.assertTrue('memory_log' in stats)
        self.assertEqual(len(stats['memory_log']), 7)  # Initial + 5 iterations + 1 forced log
        
        logger.info(f"Memory monitoring test completed, peak memory: {stats['peak_memory'] / (1024 * 1024):.2f} MB")
    
    def test_llm_engine_with_memory_monitor(self):
        """Test LLM engine integration with memory monitoring"""
        if not self.has_cuda:
            logger.info("Skipping test: No CUDA device available")
            return
            
        try:
            # Create inference engine
            logger.info("Creating inference engine...")
            engine = LLMEngine(
                model_path=self.model_path,
                config_path=self.config_path,
                tokenizer_path=self.tokenizer_path,
                device="cuda"
            )
            
            prompt = "Please provide a detailed history of artificial intelligence, from early research to modern large language models."

            # Test temperature sampling (with memory monitoring)
            logger.info(f"\nTesting temperature sampling (memory monitoring enabled):")
            temp_sampler = SamplerFactory.create_temperature_sampler(temperature=0.7)
            temp_response = engine.generate(
                prompt=prompt,
                sampler=temp_sampler,
                max_length=2048,
                memory_monitor_enabled=True,
                memory_monitor_interval=100
            )
            logger.info(f"Temperature sampling result: {temp_response}")
            
            # Validate results
            self.assertIsInstance(temp_response, str)
            self.assertTrue(len(temp_response) > 0)
            
            logger.info("LLM engine memory monitoring test passed")
            
        except Exception as e:
            logger.error(f"LLM engine memory monitoring test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"LLM engine memory monitoring test failed: {e}")
    
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