"""
Environment and Hardware Diagnostics Script.
Verifies Python version, CUDA/MPS hardware acceleration, and critical dependencies.
"""

import sys
import platform
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def check_python_version() -> bool:
    version = sys.version_info
    logging.info(f"Python Version Detected: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        return True
    logging.error("Python 3.10+ is required for this project.")
    return False

def check_pytorch_acceleration() -> None:
    try:
        import torch
        logging.info(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logging.info(f"SUCCESS: CUDA Acceleration Available! Device Count: {gpu_count}")
            logging.info(f"Primary GPU: {gpu_name}")
            logging.info(f"CUDA Version: {torch.version.cuda}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.info("SUCCESS: Apple Silicon MPS Acceleration Available!")
        else:
            logging.warning("WARNING: No GPU acceleration detected (CUDA/MPS). Models will train on CPU, which will be significantly slower.")
    except ImportError:
        logging.error("PyTorch is not installed correctly.")

def check_dependencies() -> None:
    packages = [
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("pytorch_forecasting", "PyTorch Forecasting"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("transformers", "HuggingFace Transformers"),
        ("ta", "Technical Analysis (TA) Library")
    ]
    
    logging.info("Verifying critical dependencies...")
    all_passed = True
    for pkg_name, display_name in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "Unknown")
            logging.info(f"  [✓] {display_name}: v{ver}")
        except ImportError as e:
            logging.error(f"  [✗] {display_name}: NOT INSTALLED ({e})")
            all_passed = False

    if all_passed:
        logging.info("All primary dependencies imported successfully!")
    else:
        logging.error("Some dependencies are missing. Run: pip install -r requirements.txt")

if __name__ == "__main__":
    logging.info("=== Running System Diagnostics ===")
    logging.info(f"OS: {platform.system()} {platform.release()}")
    if check_python_version():
        check_pytorch_acceleration()
        check_dependencies()
    logging.info("==================================")