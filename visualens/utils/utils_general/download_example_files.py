import os
import urllib.request
import urllib.error
from pathlib import Path
import json
import platform
from tqdm import tqdm


def _download_file_with_progress(url, file_path, file_name=None):
    """
    Download a file with a progress bar. Skips if file already exists.
    
    Parameters
    ----------
    url : str
        URL to download from
    file_path : str
        Path where to save the file
    file_name : str, optional
        Name of the file for display
    """
    if file_name is None:
        file_name = os.path.basename(file_path)
    
    # Skip if file already exists
    if os.path.exists(file_path):
        print(f"✓ {file_name} (already exists, skipping)")
        return
    
    
    def progress_hook(block_num, block_size, total_size):
        if total_size == -1:
            return
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 30
        filled = int(bar_length * downloaded / total_size)
        bar = '█' * filled + '-' * (bar_length - filled)
        print(f"\r{file_name}: |{bar}| {percent:.1f}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, file_path, progress_hook)
    print()  # New line after progress bar


def _download_directory_recursive(api_url, local_path):
    """
    Recursively download all files and folders from a GitHub directory.
    
    Parameters
    ----------
    api_url : str
        GitHub API URL for the directory
    local_path : str
        Local path where to save the files
    """
    try:
        # Fetch the list of items from GitHub API
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
        
        for item in data:
            item_name = item["name"]
            item_path = os.path.join(local_path, item_name)
            
            if item["type"] == "file":
                # Download file
                download_url = item["download_url"]
                _download_file_with_progress(download_url, item_path, item_name)
            
            elif item["type"] == "dir":
                # Create directory and recurse
                Path(item_path).mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {item_name}/")
                _download_directory_recursive(item["url"], item_path)
    
    except urllib.error.URLError as e:
        print(f"Error downloading from {api_url}: {e}")
        raise


def download_example_files(path=None):
    """
    Download example DATA files and folders from the visualens GitHub repository.
    
    Parameters
    ----------
    path : str, optional
        Path where files should be downloaded. If None, uses a default cache 
        directory based on the operating system.
    
    Returns
    -------
    str
        The path where files were downloaded.
    """
    # Define default cache directory if path is None
    if path is None:
        system = platform.system()
        if system == "Darwin":  # macOS
            cache_dir = Path.home() / "Library" / "Caches" / "visualens"
        elif system == "Windows":
            cache_dir = Path.home() / "AppData" / "Local" / "visualens" / "Cache"
        else:  # Linux and others
            cache_dir = Path.home() / ".cache" / "visualens"
        
        path = str(cache_dir)
    
    # Create the directory if it doesn't exist
    Path(path).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading example files to: {path}\n")
    
    # GitHub API endpoint to list files in the directory
    github_api_url = "https://api.github.com/repos/tomresseguier/visualens/contents/examples/DATA"
    
    # Recursively download all files and folders
    _download_directory_recursive(github_api_url, path)
    
    print(f"\n✓ Download complete! Files saved to: {path}")
    return path
    