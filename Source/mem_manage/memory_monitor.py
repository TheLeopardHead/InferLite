import torch
import logging
import time
from typing import Optional, Dict, Any, List, Union, Callable

# Create module-level logger
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """GPU memory monitor for tracking GPU memory usage"""
    
    def __init__(self, 
                 enabled: bool = True, 
                 interval: int = 100, 
                 log_level: int = logging.INFO,
                 device: Optional[Union[torch.device, int, str]] = None):
        """
        Initialize memory monitor
        Args:
            enabled: Whether to enable monitoring
            interval: Monitoring interval (number of iterations)
            log_level: Logging level
            device: Device to monitor, default is current device
        """
        self.enabled = enabled
        self.interval = interval
        self.log_level = log_level
        self.device = device
        self.iteration_count = 0
        self.start_time = None
        self.peak_memory = 0
        self.memory_log = []
        
        # Reset max memory statistics if enabled
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated(self.device)
            logger.debug(f"Memory monitor initialized: enabled={enabled}, interval={interval}")
    
    def start(self):
        """Start monitoring"""
        if not self.enabled or not torch.cuda.is_available():
            return
            
        self.start_time = time.time()
        self.iteration_count = 0
        self.peak_memory = 0
        self.memory_log = []
        
        # Record initial memory usage
        current_memory = torch.cuda.memory_allocated(self.device)
        max_memory = torch.cuda.max_memory_allocated(self.device)
        reserved_memory = torch.cuda.memory_reserved(self.device)
        
        self.memory_log.append({
            'iteration': 0,
            'time': 0,
            'current_memory': current_memory,
            'max_memory': max_memory,
            'reserved_memory': reserved_memory
        })
        
        logger.log(self.log_level, f"Memory monitoring started - Initial memory: {self._format_bytes(current_memory)}")
    
    def step(self, iteration: Optional[int] = None, force_log: bool = False):
        """
        Record memory usage for one iteration
        Args:
            iteration: Current iteration number, if None use internal counter
            force_log: Whether to force log regardless of interval
        """
        if not self.enabled or not torch.cuda.is_available():
            return
            
        # Update iteration count
        if iteration is not None:
            self.iteration_count = iteration
        else:
            self.iteration_count += 1
        
        # Check if logging is needed
        if force_log or self.iteration_count % self.interval == 0:
            current_memory = torch.cuda.memory_allocated(self.device)
            max_memory = torch.cuda.max_memory_allocated(self.device)
            reserved_memory = torch.cuda.memory_reserved(self.device)
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            # Update peak memory
            if max_memory > self.peak_memory:
                self.peak_memory = max_memory
            
            # Record memory usage
            self.memory_log.append({
                'iteration': self.iteration_count,
                'time': elapsed_time,
                'current_memory': current_memory,
                'max_memory': max_memory,
                'reserved_memory': reserved_memory
            })
            
            # Output log
            logger.log(self.log_level, 
                      f"Iteration {self.iteration_count} - "
                      f"Current memory: {self._format_bytes(current_memory)}, "
                      f"Peak memory: {self._format_bytes(max_memory)}, "
                      f"Reserved memory: {self._format_bytes(reserved_memory)}")
    
    def stop(self):
        """Stop monitoring and output statistics"""
        if not self.enabled or not torch.cuda.is_available():
            return
            
        # Record final memory usage
        current_memory = torch.cuda.memory_allocated(self.device)
        max_memory = torch.cuda.max_memory_allocated(self.device)
        reserved_memory = torch.cuda.memory_reserved(self.device)
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Output statistics
        logger.log(self.log_level, 
                  f"Memory monitoring ended - "
                  f"Total iterations: {self.iteration_count}, "
                  f"Total time: {elapsed_time:.2f}s, "
                  f"Final memory: {self._format_bytes(current_memory)}, "
                  f"Peak memory: {self._format_bytes(max_memory)}, "
                  f"Reserved memory: {self._format_bytes(reserved_memory)}")
        
        return {
            'iterations': self.iteration_count,
            'time': elapsed_time,
            'final_memory': current_memory,
            'peak_memory': max_memory,
            'reserved_memory': reserved_memory,
            'memory_log': self.memory_log
        }
    
    def _format_bytes(self, bytes_num: int) -> str:
        """Format bytes to human-readable form"""
        if bytes_num < 1024:
            return f"{bytes_num} B"
        elif bytes_num < 1024 * 1024:
            return f"{bytes_num / 1024:.2f} KB"
        elif bytes_num < 1024 * 1024 * 1024:
            return f"{bytes_num / (1024 * 1024):.2f} MB"
        else:
            return f"{bytes_num / (1024 * 1024 * 1024):.2f} GB"


def monitor_memory(func: Callable) -> Callable:
    """
    Decorator for monitoring memory usage during function execution
    Args:
        func: Function to monitor
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        # Check if memory monitoring is enabled
        memory_monitor_enabled = kwargs.pop('memory_monitor_enabled', False)
        memory_monitor_interval = kwargs.pop('memory_monitor_interval', 100)
        
        if memory_monitor_enabled and torch.cuda.is_available():
            # Create memory monitor
            monitor = MemoryMonitor(enabled=True, interval=memory_monitor_interval)
            monitor.start()
            
            # Execute original function
            try:
                result = func(*args, **kwargs)
            finally:
                # Stop monitoring and output statistics
                monitor.stop()
            
            return result
        else:
            # Execute original function without monitoring
            return func(*args, **kwargs)
    
    return wrapper 