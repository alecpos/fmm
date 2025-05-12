import os
import logging
import shutil
from huggingface_hub import hf_hub_download, login
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import psutil
from itertools import islice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_preparation.log')
    ]
)
logger = logging.getLogger(__name__)

def get_free_space(path):
    """Get free space in bytes for the given path."""
    return psutil.disk_usage(path).free

def check_disk_space(required_space_mb, path):
    """Check if there's enough disk space available."""
    free_space = get_free_space(path)
    required_space_bytes = required_space_mb * 1024 * 1024
    return free_space >= required_space_bytes, free_space / (1024 * 1024)

def download_dataset_safely(repo_id, filename, repo_type, cache_dir, hf_token=None):
    """Safely download a dataset with error handling."""
    try:
        logger.info(f"Attempting to download {filename} from {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=cache_dir,
            token=hf_token
        )
        return path
    except Exception as e:
        logger.warning(f"Could not download {filename} from {repo_id}: {str(e)}")
        logger.info("This dataset will be skipped. You can download it manually from:")
        logger.info(f"https://huggingface.co/datasets/{repo_id}")
        return None

def prepare_datasets(max_samples=None, cache_dir=None, download_mode='full', use_fiftyone=False, hf_token=None):
    """
    Prepare datasets with different download modes.
    
    Args:
        max_samples (int): Maximum number of samples to load (None for all samples)
        cache_dir (str): Custom cache directory
        download_mode (str): One of ['minimal', 'none', 'full']
            - 'minimal': Download only a few samples
            - 'none': Try to load from local cache only
            - 'full': Download full dataset
        use_fiftyone (bool): Whether to use FiftyOne for Rico dataset
        hf_token (str): Hugging Face token for authentication
    """
    try:
        # Login to Hugging Face if token is provided
        if hf_token:
            logger.info("Logging in to Hugging Face...")
            login(token=hf_token)
        
        # Create data directories
        logger.info("Creating data directories...")
        os.makedirs('pix2code_project/data/webcode2m', exist_ok=True)
        os.makedirs('pix2code_project/data/pix2code', exist_ok=True)
        os.makedirs('pix2code_project/data/rico', exist_ok=True)
        
        # Use a specific cache directory in the project folder
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), 'hf_cache')
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using project cache directory: {cache_dir}")
        
        # Check available space - require at least 5GB free for full dataset
        has_space, free_space = check_disk_space(5120, cache_dir)  # Check for 5GB
        logger.info(f"Available disk space: {free_space:.2f} MB")
        
        if not has_space:
            logger.error("Not enough disk space. Please free up at least 5GB of space.")
            return
        
        # Handle WebCode2M dataset based on download mode
        logger.info(f"Loading WebCode2M dataset in {download_mode} mode...")
        try:
            if download_mode == 'none':
                # Try to load from local cache first
                try:
                    webcode2m_dataset = load_from_disk('pix2code_project/data/webcode2m')
                    logger.info("Successfully loaded WebCode2M from local cache")
                except Exception as e:
                    logger.warning(f"Could not load from local cache: {str(e)}")
                    if has_space:
                        logger.info("Attempting full download...")
                        download_mode = 'full'
                    else:
                        raise Exception("No local cache found and insufficient disk space for download")
            
            if download_mode in ['minimal', 'full']:
                # Load dataset with streaming for memory efficiency
                webcode2m_dataset = load_dataset(
                    "xcodemind/webcode2m",
                    split="train",
                    cache_dir=cache_dir,
                    streaming=True,  # Use streaming for memory efficiency
                    token=hf_token  # Pass token for authentication
                )
                
                # Create temporary directory for intermediate saves
                temp_dir = os.path.join(cache_dir, 'temp_dataset')
                os.makedirs(temp_dir, exist_ok=True)
                
                # Process samples in batches
                batch_size = 100
                all_samples = []
                total_samples = 0
                batch_num = 0
                
                logger.info("Processing dataset in batches...")
                for sample in webcode2m_dataset:
                    all_samples.append(sample)
                    total_samples += 1
                    
                    # For minimal mode, stop after max_samples
                    if download_mode == 'minimal' and total_samples >= max_samples:
                        break
                    
                    # Save in batches to avoid memory issues
                    if len(all_samples) >= batch_size:
                        batch_dataset = Dataset.from_list(all_samples)
                        batch_path = os.path.join(temp_dir, f'batch_{batch_num}')
                        batch_dataset.save_to_disk(batch_path)
                        all_samples = []
                        batch_num += 1
                        logger.info(f"Processed {total_samples} samples so far...")
                
                # Save any remaining samples
                if all_samples:
                    batch_dataset = Dataset.from_list(all_samples)
                    batch_path = os.path.join(temp_dir, f'batch_{batch_num}')
                    batch_dataset.save_to_disk(batch_path)
                
                # Combine all batches
                logger.info("Combining batches...")
                batch_paths = [os.path.join(temp_dir, f'batch_{i}') for i in range(batch_num + 1)]
                batch_datasets = [load_from_disk(path) for path in batch_paths]
                combined_dataset = concatenate_datasets(batch_datasets)
                
                # Save final dataset
                combined_dataset.save_to_disk('pix2code_project/data/webcode2m')
                logger.info(f"Successfully processed and saved {total_samples} samples")
                
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
                
                # Load the saved dataset for logging
                webcode2m_dataset = load_from_disk('pix2code_project/data/webcode2m')
            
            # Log dataset information
            if download_mode != 'none':
                sample = webcode2m_dataset[0]
                logger.info(f"Sample data structure: {sample}")
            
        except Exception as e:
            logger.error(f"Error loading WebCode2M dataset: {str(e)}")
            raise
        
        logger.info("Dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during dataset preparation: {str(e)}")
        raise

if __name__ == '__main__':
    # Get Hugging Face token from environment variable
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        logger.warning("No Hugging Face token found. Set HF_TOKEN environment variable for full access.")
    
    # Limit to 300 samples
    prepare_datasets(
        max_samples=300,  # Limit to 300 samples
        cache_dir=None,  # Will use project directory
        download_mode='minimal',  # Use minimal mode to respect max_samples
        use_fiftyone=False,  # Disable FiftyOne by default
        hf_token=hf_token  # Pass token for authentication
    )