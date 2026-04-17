import subprocess
from typing import Optional
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base class for LLM CLI providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> Optional[str]:
        """Generate a response from the LLM CLI."""
        raise NotImplementedError
    
    def _run_cli(self, command: list, timeout: int = 300) -> Optional[str]:
        """Helper to run CLI commands via subprocess."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f"[CLI Error] {error_msg}")
                return None
            
            return result.stdout.strip()
            
        except FileNotFoundError as e:
            print(f"[CLI Error] Command not found: {command[0]}")
            return None
        except subprocess.TimeoutExpired:
            print(f"[CLI Error] Request timed out after {timeout} seconds.")
            return None
        except Exception as e:
            print(f"[CLI Exception] {e}")
            return None
