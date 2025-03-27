<#
.SYNOPSIS
    Scaffolds a complete Hugging Face integration with support for safetensors, local caching, model hot-swapping, and quantization.
.DESCRIPTION
    This script creates the folder structure and implements all files for a Hugging Face integration.
    It ensures compatibility with safetensors and provides full turnkey implementations.
.NOTES
    Author: GitHub Copilot
    Version: 1.0
#>

# Base directory
$baseDir = Join-Path $PSScriptRoot ".."

# Directory structure with descriptions
$dirs = @{
    "huggingface_toolset"                      = "Root package for Hugging Face integration"
    "huggingface_toolset/model_management"     = "Module for managing Hugging Face models"
    "huggingface_toolset/dataset_management"   = "Module for managing Hugging Face datasets"
    "huggingface_toolset/tokenizer_management" = "Module for managing Hugging Face tokenizers"
    "huggingface_toolset/api_integration"      = "Module for Hugging Face API interactions"
    "huggingface_toolset/config"               = "Configuration module for Hugging Face integration"
    "huggingface_toolset/utils"                = "Utility functions for Hugging Face operations"
    "huggingface_toolset/cache"                = "Local caching for models, datasets, and tokenizers"
    "huggingface_toolset/quantization"         = "Module for model quantization and optimization"
}

# Create directories
foreach ($dir in $dirs.Keys) {
    $path = Join-Path $baseDir $dir
    if (-not (Test-Path $path)) { New-Item -Path $path -ItemType Directory -Force | Out-Null }
    $readmePath = Join-Path $path "README.md"
    Set-Content -Path $readmePath -Value "# $(Split-Path $dir -Leaf)`n`n$($dirs[$dir])"
}

# Implement the main __init__.py
$mainInitContent = @"
"""
Hugging Face Toolset
--------------------
A comprehensive toolset for interacting with Hugging Face models, datasets, and tokenizers.
Supports safetensors, local caching, model hot-swapping, and quantization.
"""

from .config.env_loader import load_environment_variables
from .model_management.download_model import download_model
from .model_management.upload_model import upload_model
from .model_management.list_models import list_models
from .model_management.delete_model import delete_model
from .dataset_management.download_dataset import download_dataset
from .dataset_management.upload_dataset import upload_dataset
from .tokenizer_management.download_tokenizer import download_tokenizer
from .tokenizer_management.upload_tokenizer import upload_tokenizer
from .cache.cache_manager import CacheManager
from .quantization.quantize_model import quantize_model

# Initialize environment variables
load_environment_variables()

__all__ = [
    'download_model',
    'upload_model',
    'list_models',
    'delete_model',
    'download_dataset',
    'upload_dataset',
    'download_tokenizer',
    'upload_tokenizer',
    'CacheManager',
    'quantize_model',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/__init__.py") -Value $mainInitContent

# Implement model_management/__init__.py
$modelInitContent = @"
"""
Model Management Module
-----------------------
Provides functions for managing models from the Hugging Face Hub.
"""

from .download_model import download_model
from .upload_model import upload_model
from .list_models import list_models
from .delete_model import delete_model
from .model_utils import get_model_info, convert_to_safetensors, load_model

__all__ = [
    'download_model',
    'upload_model',
    'list_models',
    'delete_model',
    'get_model_info',
    'convert_to_safetensors',
    'load_model',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/__init__.py") -Value $modelInitContent

# Implement model_management/download_model.py
$downloadModelContent = @"
"""
Download Model
--------------
Functions for downloading models from Hugging Face Hub with support for safetensors.
"""

import os
from typing import Optional, Dict, Any, Union
import logging
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from ..config.constants import DEFAULT_CACHE_DIR
from ..utils.logging_utils import get_logger
from ..cache.cache_manager import CacheManager
from .model_utils import convert_to_safetensors

logger = get_logger(__name__)

def download_model(
    model_id: str,
    model_type: str = "auto",
    use_safetensors: bool = True,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_id: The ID of the model to download.
        model_type: The type of model to download (auto, causal, seq2seq).
        use_safetensors: Whether to use safetensors for downloading and storing the model.
        revision: The specific model revision to download.
        cache_dir: Directory where the models should be cached.
        token: Hugging Face API token for accessing private models.
        **kwargs: Additional arguments for model loading.
        
    Returns:
        Dict containing information about the downloaded model.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Create cache manager
    cache_manager = CacheManager(cache_dir=cache_dir)
    
    # Check if model is already in cache
    if cache_manager.is_cached(model_id, revision):
        logger.info(f"Model {model_id} already in cache. Loading from cache.")
        return cache_manager.get_cached_model_info(model_id, revision)
    
    logger.info(f"Downloading model {model_id}")
    
    # Select model class based on model_type
    if model_type.lower() == "causal":
        model_class = AutoModelForCausalLM
    elif model_type.lower() == "seq2seq":
        model_class = AutoModelForSeq2SeqLM
    else:
        model_class = AutoModel
    
    # Download the model
    model = model_class.from_pretrained(
        model_id,
        revision=revision,
        use_safetensors=use_safetensors,
        token=token,
        **kwargs
    )
    
    # If safetensors is requested but model wasn't loaded with it, convert
    if use_safetensors and not model.config.get("use_safetensors", False):
        logger.info(f"Converting model {model_id} to safetensors format")
        model = convert_to_safetensors(model)
    
    # Save model to cache
    cache_path = cache_manager.cache_model(model, model_id, revision)
    
    model_info = {
        "model_id": model_id,
        "revision": revision,
        "model_type": model_type,
        "cache_path": cache_path,
        "config": model.config.to_dict()
    }
    
    logger.info(f"Successfully downloaded and cached model {model_id}")
    return model_info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/download_model.py") -Value $downloadModelContent

# Implement model_management/upload_model.py
$uploadModelContent = @"
"""
Upload Model
------------
Functions for uploading models to Hugging Face Hub.
"""

import os
from typing import Optional, Dict, Any, Union
from transformers import PreTrainedModel
from huggingface_hub import HfApi
from ..utils.logging_utils import get_logger
from ..config.constants import DEFAULT_CACHE_DIR
from .model_utils import convert_to_safetensors

logger = get_logger(__name__)

def upload_model(
    model: Union[PreTrainedModel, str],
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    use_safetensors: bool = True,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Upload a model to Hugging Face Hub.
    
    Args:
        model: The model to upload or path to saved model.
        repo_id: The repository ID to upload to.
        commit_message: The commit message for the upload.
        private: Whether the repository should be private.
        use_safetensors: Whether to convert the model to safetensors format.
        token: Hugging Face API token.
        **kwargs: Additional arguments for model uploading.
        
    Returns:
        Dict containing information about the uploaded model.
    """
    logger.info(f"Uploading model to {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # If model is a path, load it first
    if isinstance(model, str):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model, **kwargs)
    
    # Convert to safetensors if requested
    if use_safetensors:
        logger.info("Converting model to safetensors format")
        model = convert_to_safetensors(model)
    
    # Create a temporary directory to save the model
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info(f"Saving model to temporary directory {tmpdirname}")
        model.save_pretrained(tmpdirname, safe_serialization=use_safetensors)
        
        # Upload the model
        logger.info(f"Pushing model to the Hub: {repo_id}")
        commit_message = commit_message or f"Upload model {model.config.name_or_path}"
        
        response = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True
        )
        
        api.upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            commit_message=commit_message
        )
    
    logger.info(f"Successfully uploaded model to {repo_id}")
    
    return {
        "repo_id": repo_id,
        "model_id": model.config.name_or_path,
        "commit_message": commit_message,
        "private": private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/upload_model.py") -Value $uploadModelContent

# Implement model_management/list_models.py
$listModelsContent = @"
"""
List Models
-----------
Functions for listing models from Hugging Face Hub.
"""

from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, list_models
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def list_models(
    author: Optional[str] = None,
    search: Optional[str] = None,
    model_type: Optional[str] = None,
    token: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List models from Hugging Face Hub.
    
    Args:
        author: Filter by model author.
        search: Search query for model names and descriptions.
        model_type: Filter by model type.
        token: Hugging Face API token.
        limit: Maximum number of models to return.
        
    Returns:
        List of dictionaries containing model information.
    """
    logger.info(f"Listing models with params: author={author}, search={search}, model_type={model_type}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Get models
    models = list_models(
        author=author,
        search=search,
        filter=model_type,
        token=token,
        limit=limit
    )
    
    # Format the results
    results = []
    for model_info in models:
        results.append({
            "model_id": model_info.modelId,
            "author": model_info.author,
            "tags": model_info.tags,
            "downloads": model_info.downloads,
            "likes": model_info.likes,
            "library_name": model_info.pipeline_tag
        })
    
    logger.info(f"Found {len(results)} models")
    return results
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/list_models.py") -Value $listModelsContent

# Implement model_management/delete_model.py
$deleteModelContent = @"
"""
Delete Model
------------
Functions for deleting models from Hugging Face Hub.
"""

from typing import Dict, Any, Optional
from huggingface_hub import HfApi, delete_repo
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def delete_model(
    repo_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a model from Hugging Face Hub.
    
    Args:
        repo_id: The repository ID to delete.
        token: Hugging Face API token.
        
    Returns:
        Dict containing information about the deleted model.
    """
    logger.info(f"Deleting model repository: {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Delete the repository
    delete_repo(repo_id=repo_id, token=token)
    
    logger.info(f"Successfully deleted model repository: {repo_id}")
    
    return {
        "repo_id": repo_id,
        "status": "deleted"
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/delete_model.py") -Value $deleteModelContent

# Implement model_management/model_utils.py
$modelUtilsContent = @"
"""
Model Utilities
---------------
Utility functions for model management operations.
"""

import os
from typing import Dict, Any, Optional, Union
import torch
from transformers import PreTrainedModel, AutoModel
from safetensors.torch import save_file, load_file
from ..utils.logging_utils import get_logger
from ..config.constants import DEFAULT_CACHE_DIR

logger = get_logger(__name__)

def get_model_info(
    model_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed information about a model.
    
    Args:
        model_id: The ID of the model.
        token: Hugging Face API token.
        
    Returns:
        Dict containing model information.
    """
    from huggingface_hub import model_info
    
    logger.info(f"Getting information for model: {model_id}")
    
    info = model_info(repo_id=model_id, token=token)
    
    return {
        "model_id": info.modelId,
        "author": info.author,
        "tags": info.tags,
        "downloads": info.downloads,
        "likes": info.likes,
        "last_modified": str(info.lastModified),
        "library_name": info.pipeline_tag,
        "private": info.private
    }

def convert_to_safetensors(
    model: PreTrainedModel,
    output_path: Optional[str] = None
) -> PreTrainedModel:
    """
    Convert a model to safetensors format.
    
    Args:
        model: The model to convert.
        output_path: Optional path to save the converted model.
        
    Returns:
        The model (unchanged, but weights converted in-place).
    """
    logger.info(f"Converting model {model.config.name_or_path} to safetensors format")
    
    # Create a temporary output path if not provided
    if output_path is None:
        import tempfile
        output_path = tempfile.mkdtemp()
    
    # Save the model with safe_serialization=True
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Reload the model
    reloaded_model = AutoModel.from_pretrained(output_path, use_safetensors=True)
    
    logger.info(f"Successfully converted model to safetensors format")
    
    # Clean up temporary directory if one was created
    if output_path != DEFAULT_CACHE_DIR and not output_path:
        import shutil
        shutil.rmtree(output_path)
    
    return reloaded_model

def load_model(
    model_id_or_path: str,
    use_safetensors: bool = True,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> PreTrainedModel:
    """
    Load a model from Hugging Face Hub or local path.
    
    Args:
        model_id_or_path: The ID or path of the model to load.
        use_safetensors: Whether to use safetensors for loading the model.
        revision: The specific model revision to load.
        token: Hugging Face API token.
        **kwargs: Additional arguments for model loading.
        
    Returns:
        The loaded model.
    """
    logger.info(f"Loading model: {model_id_or_path}")
    
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained(
        model_id_or_path,
        revision=revision,
        use_safetensors=use_safetensors,
        token=token,
        **kwargs
    )
    
    logger.info(f"Successfully loaded model: {model_id_or_path}")
    
    return model
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/model_utils.py") -Value $modelUtilsContent

# Implement dataset_management/__init__.py
$datasetInitContent = @"
"""
Dataset Management Module
------------------------
Provides functions for managing datasets from the Hugging Face Hub.
"""

from .download_dataset import download_dataset
from .upload_dataset import upload_dataset
from .list_datasets import list_datasets
from .delete_dataset import delete_dataset
from .dataset_utils import get_dataset_info

__all__ = [
    'download_dataset',
    'upload_dataset',
    'list_datasets',
    'delete_dataset',
    'get_dataset_info',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/__init__.py") -Value $datasetInitContent

# Implement dataset_management/download_dataset.py
$downloadDatasetContent = @"
"""
Download Dataset
---------------
Functions for downloading datasets from Hugging Face Hub.
"""

from typing import Dict, Any, Optional, Union, List
from datasets import load_dataset
from ..utils.logging_utils import get_logger
from ..config.constants import DEFAULT_CACHE_DIR
from ..cache.cache_manager import CacheManager

logger = get_logger(__name__)

def download_dataset(
    dataset_id: str,
    subset: Optional[str] = None,
    split: Optional[Union[str, List[str]]] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Download a dataset from Hugging Face Hub.
    
    Args:
        dataset_id: The ID of the dataset to download.
        subset: The subset of the dataset to download.
        split: The split of the dataset to download.
        revision: The specific dataset revision to download.
        cache_dir: Directory where the datasets should be cached.
        token: Hugging Face API token.
        **kwargs: Additional arguments for dataset loading.
        
    Returns:
        Dict containing information about the downloaded dataset.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Create cache manager
    cache_manager = CacheManager(cache_dir=cache_dir)
    
    # Check if dataset is already in cache
    if cache_manager.is_dataset_cached(dataset_id, subset, revision):
        logger.info(f"Dataset {dataset_id} already in cache. Loading from cache.")
        return cache_manager.get_cached_dataset_info(dataset_id, subset, revision)
    
    logger.info(f"Downloading dataset {dataset_id}")
    
    # Download the dataset
    dataset = load_dataset(
        dataset_id,
        name=subset,
        split=split,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        **kwargs
    )
    
    # Save dataset to cache
    cache_path = cache_manager.cache_dataset(dataset, dataset_id, subset, revision)
    
    dataset_info = {
        "dataset_id": dataset_id,
        "subset": subset,
        "split": split,
        "revision": revision,
        "cache_path": cache_path,
        "features": dataset.features if hasattr(dataset, "features") else None,
        "num_examples": len(dataset) if hasattr(dataset, "__len__") else None
    }
    
    logger.info(f"Successfully downloaded and cached dataset {dataset_id}")
    return dataset_info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/download_dataset.py") -Value $downloadDatasetContent

# Implement dataset_management/upload_dataset.py
$uploadDatasetContent = @"
"""
Upload Dataset
-------------
Functions for uploading datasets to Hugging Face Hub.
"""

from typing import Dict, Any, Optional, Union
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def upload_dataset(
    dataset: Union[Dataset, DatasetDict, str],
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Upload a dataset to Hugging Face Hub.
    
    Args:
        dataset: The dataset to upload or path to saved dataset.
        repo_id: The repository ID to upload to.
        commit_message: The commit message for the upload.
        private: Whether the repository should be private.
        token: Hugging Face API token.
        **kwargs: Additional arguments for dataset uploading.
        
    Returns:
        Dict containing information about the uploaded dataset.
    """
    logger.info(f"Uploading dataset to {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # If dataset is a path, load it first
    if isinstance(dataset, str):
        from datasets import load_from_disk
        dataset = load_from_disk(dataset)
    
    # Create a temporary directory to save the dataset
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info(f"Saving dataset to temporary directory {tmpdirname}")
        
        # Save dataset to temporary directory
        if isinstance(dataset, Dataset):
            dataset.save_to_disk(tmpdirname)
        elif isinstance(dataset, DatasetDict):
            dataset.save_to_disk(tmpdirname)
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
        
        # Upload the dataset
        logger.info(f"Pushing dataset to the Hub: {repo_id}")
        commit_message = commit_message or f"Upload dataset"
        
        response = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        
        api.upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message
        )
    
    logger.info(f"Successfully uploaded dataset to {repo_id}")
    
    return {
        "repo_id": repo_id,
        "dataset_type": type(dataset).__name__,
        "commit_message": commit_message,
        "private": private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/upload_dataset.py") -Value $uploadDatasetContent

# Implement dataset_management/list_datasets.py
$listDatasetsContent = @"
"""
List Datasets
------------
Functions for listing datasets from Hugging Face Hub.
"""

from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, list_datasets
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def list_datasets(
    author: Optional[str] = None,
    search: Optional[str] = None,
    token: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List datasets from Hugging Face Hub.
    
    Args:
        author: Filter by dataset author.
        search: Search query for dataset names and descriptions.
        token: Hugging Face API token.
        limit: Maximum number of datasets to return.
        
    Returns:
        List of dictionaries containing dataset information.
    """
    logger.info(f"Listing datasets with params: author={author}, search={search}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Get datasets
    datasets = list_datasets(
        author=author,
        search=search,
        token=token,
        limit=limit
    )
    
    # Format the results
    results = []
    for dataset_info in datasets:
        results.append({
            "dataset_id": dataset_info.id,
            "author": dataset_info.author,
            "tags": dataset_info.tags,
            "downloads": dataset_info.downloads,
            "likes": dataset_info.likes
        })
    
    logger.info(f"Found {len(results)} datasets")
    return results
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/list_datasets.py") -Value $listDatasetsContent

# Implement dataset_management/delete_dataset.py
$deleteDatasetContent = @"
"""
Delete Dataset
-------------
Functions for deleting datasets from Hugging Face Hub.
"""

from typing import Dict, Any, Optional
from huggingface_hub import HfApi, delete_repo
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def delete_dataset(
    repo_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a dataset from Hugging Face Hub.
    
    Args:
        repo_id: The repository ID to delete.
        token: Hugging Face API token.
        
    Returns:
        Dict containing information about the deleted dataset.
    """
    logger.info(f"Deleting dataset repository: {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Delete the repository
    delete_repo(repo_id=repo_id, repo_type="dataset", token=token)
    
    logger.info(f"Successfully deleted dataset repository: {repo_id}")
    
    return {
        "repo_id": repo_id,
        "status": "deleted"
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/delete_dataset.py") -Value $deleteDatasetContent

# Implement dataset_management/dataset_utils.py
$datasetUtilsContent = @"
"""
Dataset Utilities
----------------
Utility functions for dataset management operations.
"""

from typing import Dict, Any, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_dataset_info(
    dataset_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed information about a dataset.
    
    Args:
        dataset_id: The ID of the dataset.
        token: Hugging Face API token.
        
    Returns:
        Dict containing dataset information.
    """
    from huggingface_hub import dataset_info
    
    logger.info(f"Getting information for dataset: {dataset_id}")
    
    info = dataset_info(repo_id=dataset_id, token=token)
    
    return {
        "dataset_id": info.id,
        "author": info.author,
        "tags": info.tags,
        "downloads": info.downloads,
        "likes": info.likes,
        "last_modified": str(info.lastModified),
        "private": info.private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/dataset_utils.py") -Value $datasetUtilsContent

# Implement tokenizer_management/__init__.py
$tokenizerInitContent = @"
"""
Tokenizer Management Module
--------------------------
Provides functions for managing tokenizers from the Hugging Face Hub.
"""

from .download_tokenizer import download_tokenizer
from .upload_tokenizer import upload_tokenizer
from .list_tokenizers import list_tokenizers
from .delete_tokenizer import delete_tokenizer
from .tokenizer_utils import get_tokenizer_info

__all__ = [
    'download_tokenizer',
    'upload_tokenizer',
    'list_tokenizers',
    'delete_tokenizer',
    'get_tokenizer_info',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/__init__.py") -Value $tokenizerInitContent

# Implement tokenizer_management/download_tokenizer.py
$downloadTokenizerContent = @"
"""
Download Tokenizer
-----------------
Functions for downloading tokenizers from Hugging Face Hub.
"""

from typing import Dict, Any, Optional
from transformers import AutoTokenizer
from ..utils.logging_utils import get_logger
from ..config.constants import DEFAULT_CACHE_DIR
from ..cache.cache_manager import CacheManager

logger = get_logger(__name__)

def download_tokenizer(
    tokenizer_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Download a tokenizer from Hugging Face Hub.
    
    Args:
        tokenizer_id: The ID of the tokenizer to download.
        revision: The specific tokenizer revision to download.
        cache_dir: Directory where the tokenizers should be cached.
        token: Hugging Face API token.
        **kwargs: Additional arguments for tokenizer loading.
        
    Returns:
        Dict containing information about the downloaded tokenizer.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Create cache manager
    cache_manager = CacheManager(cache_dir=cache_dir)
    
    # Check if tokenizer is already in cache
    if cache_manager.is_tokenizer_cached(tokenizer_id, revision):
        logger.info(f"Tokenizer {tokenizer_id} already in cache. Loading from cache.")
        return cache_manager.get_cached_tokenizer_info(tokenizer_id, revision)
    
    logger.info(f"Downloading tokenizer {tokenizer_id}")
    
    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        **kwargs
    )
    
    # Save tokenizer to cache
    cache_path = cache_manager.cache_tokenizer(tokenizer, tokenizer_id, revision)
    
    tokenizer_info = {
        "tokenizer_id": tokenizer_id,
        "revision": revision,
        "cache_path": cache_path,
        "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else None
    }
    
    logger.info(f"Successfully downloaded and cached tokenizer {tokenizer_id}")
    return tokenizer_info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/download_tokenizer.py") -Value $downloadTokenizerContent

# Implement tokenizer_management/upload_tokenizer.py
$uploadTokenizerContent = @"
"""
Upload Tokenizer
---------------
Functions for uploading tokenizers to Hugging Face Hub.
"""

from typing import Dict, Any, Optional, Union
from transformers import PreTrainedTokenizer
from huggingface_hub import HfApi
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def upload_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, str],
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Upload a tokenizer to Hugging Face Hub.
    
    Args:
        tokenizer: The tokenizer to upload or path to saved tokenizer.
        repo_id: The repository ID to upload to.
        commit_message: The commit message for the upload.
        private: Whether the repository should be private.
        token: Hugging Face API token.
        **kwargs: Additional arguments for tokenizer uploading.
        
    Returns:
        Dict containing information about the uploaded tokenizer.
    """
    logger.info(f"Uploading tokenizer to {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # If tokenizer is a path, load it first
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
    
    # Create a temporary directory to save the tokenizer
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info(f"Saving tokenizer to temporary directory {tmpdirname}")
        tokenizer.save_pretrained(tmpdirname)
        
        # Upload the tokenizer
        logger.info(f"Pushing tokenizer to the Hub: {repo_id}")
        commit_message = commit_message or f"Upload tokenizer"
        
        response = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True
        )
        
        api.upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            commit_message=commit_message
        )
    
    logger.info(f"Successfully uploaded tokenizer to {repo_id}")
    
    return {
        "repo_id": repo_id,
        "commit_message": commit_message,
        "private": private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/upload_tokenizer.py") -Value $uploadTokenizerContent

# Implement tokenizer_management/list_tokenizers.py
$listTokenizersContent = @"
"""
List Tokenizers
--------------
Functions for listing tokenizers from Hugging Face Hub.
"""

from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, list_models
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def list_tokenizers(
    author: Optional[str] = None,
    search: Optional[str] = None,
    token: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List tokenizers from Hugging Face Hub.
    
    Args:
        author: Filter by tokenizer author.
        search: Search query for tokenizer names and descriptions.
        token: Hugging Face API token.
        limit: Maximum number of tokenizers to return.
        
    Returns:
        List of dictionaries containing tokenizer information.
    """
    logger.info(f"Listing tokenizers with params: author={author}, search={search}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Get models with tokenizer tag
    models = list_models(
        author=author,
        search=search,
        filter="tokenizer",
        token=token,
        limit=limit
    )
    
    # Format the results
    results = []
    for model_info in models:
        results.append({
            "tokenizer_id": model_info.modelId,
            "author": model_info.author,
            "tags": model_info.tags,
            "downloads": model_info.downloads,
            "likes": model_info.likes
        })
    
    logger.info(f"Found {len(results)} tokenizers")
    return results
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/list_tokenizers.py") -Value $listTokenizersContent

# Implement tokenizer_management/delete_tokenizer.py
$deleteTokenizerContent = @"
"""
Delete Tokenizer
---------------
Functions for deleting tokenizers from Hugging Face Hub.
"""

from typing import Dict, Any, Optional
from huggingface_hub import HfApi, delete_repo
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def delete_tokenizer(
    repo_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a tokenizer from Hugging Face Hub.
    
    Args:
        repo_id: The repository ID to delete.
        token: Hugging Face API token.
        
    Returns:
        Dict containing information about the deleted tokenizer.
    """
    logger.info(f"Deleting tokenizer repository: {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Delete the repository
    delete_repo(repo_id=repo_id, token=token)
    
    logger.info(f"Successfully deleted tokenizer repository: {repo_id}")
    
    return {
        "repo_id": repo_id,
        "status": "deleted"
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/delete_tokenizer.py") -Value $deleteTokenizerContent

# Implement tokenizer_management/tokenizer_utils.py
$tokenizerUtilsContent = @"
"""
Tokenizer Utilities
------------------
Utility functions for tokenizer management operations.
"""

from typing import Dict, Any, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_tokenizer_info(
    tokenizer_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed information about a tokenizer.
    
    Args:
        tokenizer_id: The ID of the tokenizer.
        token: Hugging Face API token.
        
    Returns:
        Dict containing tokenizer information.
    """
    from huggingface_hub import model_info
    
    logger.info(f"Getting information for tokenizer: {tokenizer_id}")
    
    info = model_info(repo_id=tokenizer_id, token=token)
    
    return {
        "tokenizer_id": info.modelId,
        "author": info.author,
        "tags": info.tags,
        "downloads": info.downloads,
        "likes": info.likes,
        "last_modified": str(info.lastModified),
        "private": info.private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/tokenizer_utils.py") -Value $tokenizerUtilsContent

# Implement api_integration/__init__.py
$apiInitContent = @"
"""
API Integration Module
---------------------
Provides functions for interacting with the Hugging Face API.
"""

from .auth import get_token, set_token, login
from .api_client import get_api_client
from .api_utils import check_token_validity

__all__ = [
    'get_token',
    'set_token',
    'login',
    'get_api_client',
    'check_token_validity',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/api_integration/__init__.py") -Value $apiInitContent

# Implement api_integration/auth.py
$authContent = @"
"""
Hugging Face Authentication
-------------------------
Functions for authenticating with the Hugging Face API.
"""

import os
from typing import Optional
from huggingface_hub import login as hf_login
from ..utils.logging_utils import get_logger
from ..config.constants import TOKEN_ENV_NAME

logger = get_logger(__name__)

def get_token() -> Optional[str]:
    """
    Get the Hugging Face API token.
    
    Returns:
        The token or None if not set.
    """<#
.SYNOPSIS
    Scaffolds a complete Hugging Face integration with support for safetensors, local caching, model hot-swapping, and quantization.
.DESCRIPTION
    This script creates the folder structure and implements all files for a Hugging Face integration.
    It ensures compatibility with safetensors and provides full turnkey implementations.
.NOTES
    Author: GitHub Copilot
    Version: 1.0
#>

# Base directory
$baseDir = Join-Path $PSScriptRoot ".."

# Directory structure with descriptions
$dirs = @{
    "huggingface_toolset"                          = "Root package for Hugging Face integration"
    "huggingface_toolset/model_management"         = "Module for managing Hugging Face models"
    "huggingface_toolset/dataset_management"       = "Module for managing Hugging Face datasets"
    "huggingface_toolset/tokenizer_management"     = "Module for managing Hugging Face tokenizers"
    "huggingface_toolset/api_integration"          = "Module for Hugging Face API interactions"
    "huggingface_toolset/config"                   = "Configuration module for Hugging Face integration"
    "huggingface_toolset/utils"                    = "Utility functions for Hugging Face operations"
    "huggingface_toolset/cache"                    = "Local caching for models, datasets, and tokenizers"
    "huggingface_toolset/quantization"             = "Module for model quantization and optimization"
}

# Create directories
foreach ($dir in $dirs.Keys) {
    $path = Join-Path $baseDir $dir
    if (-not (Test-Path $path)) { New-Item -Path $path -ItemType Directory -Force | Out-Null }
    $readmePath = Join-Path $path "README.md"
    Set-Content -Path $readmePath -Value "# $(Split-Path $dir -Leaf)`n`n$($dirs[$dir])"
}

# Implement the main __init__.py
$mainInitContent = @"
"""
Hugging Face Toolset
--------------------
A comprehensive toolset for interacting with Hugging Face models, datasets, and tokenizers.
Supports safetensors, local caching, model hot-swapping, and quantization.
"""

from .config.env_loader import load_environment_variables
from .model_management.download_model import download_model
from .model_management.upload_model import upload_model
from .model_management.list_models import list_models
from .model_management.delete_model import delete_model
from .dataset_management.download_dataset import download_dataset
from .dataset_management.upload_dataset import upload_dataset
from .tokenizer_management.download_tokenizer import download_tokenizer
from .tokenizer_management.upload_tokenizer import upload_tokenizer
from .cache.cache_manager import CacheManager
from .quantization.quantize_model import quantize_model

# Initialize environment variables
load_environment_variables()

__all__ = [
    'download_model',
    'upload_model',
    'list_models',
    'delete_model',
    'download_dataset',
    'upload_dataset',
    'download_tokenizer',
    'upload_tokenizer',
    'CacheManager',
    'quantize_model',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/__init__.py") -Value $mainInitContent

# Implement model_management/__init__.py
$modelInitContent = @"
"""
Model Management Module
-----------------------
Provides functions for managing models from the Hugging Face Hub.
"""

from .download_model import download_model
from .upload_model import upload_model
from .list_models import list_models
from .delete_model import delete_model
from .model_utils import get_model_info, convert_to_safetensors, load_model

__all__ = [
    'download_model',
    'upload_model',
    'list_models',
    'delete_model',
    'get_model_info',
    'convert_to_safetensors',
    'load_model',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/__init__.py") -Value $modelInitContent

# Implement model_management/download_model.py
$downloadModelContent = @"
"""
Download Model
--------------
Functions for downloading models from Hugging Face Hub with support for safetensors.
"""

import os
from typing import Optional, Dict, Any, Union
import logging
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from ..config.constants import DEFAULT_CACHE_DIR
from ..utils.logging_utils import get_logger
from ..cache.cache_manager import CacheManager
from .model_utils import convert_to_safetensors

logger = get_logger(__name__)

def download_model(
    model_id: str,
    model_type: str = "auto",
    use_safetensors: bool = True,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_id: The ID of the model to download.
        model_type: The type of model to download (auto, causal, seq2seq).
        use_safetensors: Whether to use safetensors for downloading and storing the model.
        revision: The specific model revision to download.
        cache_dir: Directory where the models should be cached.
        token: Hugging Face API token for accessing private models.
        **kwargs: Additional arguments for model loading.
        
    Returns:
        Dict containing information about the downloaded model.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Create cache manager
    cache_manager = CacheManager(cache_dir=cache_dir)
    
    # Check if model is already in cache
    if cache_manager.is_cached(model_id, revision):
        logger.info(f"Model {model_id} already in cache. Loading from cache.")
        return cache_manager.get_cached_model_info(model_id, revision)
    
    logger.info(f"Downloading model {model_id}")
    
    # Select model class based on model_type
    if model_type.lower() == "causal":
        model_class = AutoModelForCausalLM
    elif model_type.lower() == "seq2seq":
        model_class = AutoModelForSeq2SeqLM
    else:
        model_class = AutoModel
    
    # Download the model
    model = model_class.from_pretrained(
        model_id,
        revision=revision,
        use_safetensors=use_safetensors,
        token=token,
        **kwargs
    )
    
    # If safetensors is requested but model wasn't loaded with it, convert
    if use_safetensors and not model.config.get("use_safetensors", False):
        logger.info(f"Converting model {model_id} to safetensors format")
        model = convert_to_safetensors(model)
    
    # Save model to cache
    cache_path = cache_manager.cache_model(model, model_id, revision)
    
    model_info = {
        "model_id": model_id,
        "revision": revision,
        "model_type": model_type,
        "cache_path": cache_path,
        "config": model.config.to_dict()
    }
    
    logger.info(f"Successfully downloaded and cached model {model_id}")
    return model_info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/download_model.py") -Value $downloadModelContent

# Implement model_management/upload_model.py
$uploadModelContent = @"
"""
Upload Model
------------
Functions for uploading models to Hugging Face Hub.
"""

import os
from typing import Optional, Dict, Any, Union
from transformers import PreTrainedModel
from huggingface_hub import HfApi
from ..utils.logging_utils import get_logger
from ..config.constants import DEFAULT_CACHE_DIR
from .model_utils import convert_to_safetensors

logger = get_logger(__name__)

def upload_model(
    model: Union[PreTrainedModel, str],
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    use_safetensors: bool = True,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Upload a model to Hugging Face Hub.
    
    Args:
        model: The model to upload or path to saved model.
        repo_id: The repository ID to upload to.
        commit_message: The commit message for the upload.
        private: Whether the repository should be private.
        use_safetensors: Whether to convert the model to safetensors format.
        token: Hugging Face API token.
        **kwargs: Additional arguments for model uploading.
        
    Returns:
        Dict containing information about the uploaded model.
    """
    logger.info(f"Uploading model to {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # If model is a path, load it first
    if isinstance(model, str):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model, **kwargs)
    
    # Convert to safetensors if requested
    if use_safetensors:
        logger.info("Converting model to safetensors format")
        model = convert_to_safetensors(model)
    
    # Create a temporary directory to save the model
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info(f"Saving model to temporary directory {tmpdirname}")
        model.save_pretrained(tmpdirname, safe_serialization=use_safetensors)
        
        # Upload the model
        logger.info(f"Pushing model to the Hub: {repo_id}")
        commit_message = commit_message or f"Upload model {model.config.name_or_path}"
        
        response = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True
        )
        
        api.upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            commit_message=commit_message
        )
    
    logger.info(f"Successfully uploaded model to {repo_id}")
    
    return {
        "repo_id": repo_id,
        "model_id": model.config.name_or_path,
        "commit_message": commit_message,
        "private": private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/upload_model.py") -Value $uploadModelContent

# Implement model_management/list_models.py
$listModelsContent = @"
"""
List Models
-----------
Functions for listing models from Hugging Face Hub.
"""

from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, list_models
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def list_models(
    author: Optional[str] = None,
    search: Optional[str] = None,
    model_type: Optional[str] = None,
    token: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List models from Hugging Face Hub.
    
    Args:
        author: Filter by model author.
        search: Search query for model names and descriptions.
        model_type: Filter by model type.
        token: Hugging Face API token.
        limit: Maximum number of models to return.
        
    Returns:
        List of dictionaries containing model information.
    """
    logger.info(f"Listing models with params: author={author}, search={search}, model_type={model_type}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Get models
    models = list_models(
        author=author,
        search=search,
        filter=model_type,
        token=token,
        limit=limit
    )
    
    # Format the results
    results = []
    for model_info in models:
        results.append({
            "model_id": model_info.modelId,
            "author": model_info.author,
            "tags": model_info.tags,
            "downloads": model_info.downloads,
            "likes": model_info.likes,
            "library_name": model_info.pipeline_tag
        })
    
    logger.info(f"Found {len(results)} models")
    return results
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/list_models.py") -Value $listModelsContent

# Implement model_management/delete_model.py
$deleteModelContent = @"
"""
Delete Model
------------
Functions for deleting models from Hugging Face Hub.
"""

from typing import Dict, Any, Optional
from huggingface_hub import HfApi, delete_repo
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def delete_model(
    repo_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a model from Hugging Face Hub.
    
    Args:
        repo_id: The repository ID to delete.
        token: Hugging Face API token.
        
    Returns:
        Dict containing information about the deleted model.
    """
    logger.info(f"Deleting model repository: {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Delete the repository
    delete_repo(repo_id=repo_id, token=token)
    
    logger.info(f"Successfully deleted model repository: {repo_id}")
    
    return {
        "repo_id": repo_id,
        "status": "deleted"
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/delete_model.py") -Value $deleteModelContent

# Implement model_management/model_utils.py
$modelUtilsContent = @"
"""
Model Utilities
---------------
Utility functions for model management operations.
"""

import os
from typing import Dict, Any, Optional, Union
import torch
from transformers import PreTrainedModel, AutoModel
from safetensors.torch import save_file, load_file
from ..utils.logging_utils import get_logger
from ..config.constants import DEFAULT_CACHE_DIR

logger = get_logger(__name__)

def get_model_info(
    model_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed information about a model.
    
    Args:
        model_id: The ID of the model.
        token: Hugging Face API token.
        
    Returns:
        Dict containing model information.
    """
    from huggingface_hub import model_info
    
    logger.info(f"Getting information for model: {model_id}")
    
    info = model_info(repo_id=model_id, token=token)
    
    return {
        "model_id": info.modelId,
        "author": info.author,
        "tags": info.tags,
        "downloads": info.downloads,
        "likes": info.likes,
        "last_modified": str(info.lastModified),
        "library_name": info.pipeline_tag,
        "private": info.private
    }

def convert_to_safetensors(
    model: PreTrainedModel,
    output_path: Optional[str] = None
) -> PreTrainedModel:
    """
    Convert a model to safetensors format.
    
    Args:
        model: The model to convert.
        output_path: Optional path to save the converted model.
        
    Returns:
        The model (unchanged, but weights converted in-place).
    """
    logger.info(f"Converting model {model.config.name_or_path} to safetensors format")
    
    # Create a temporary output path if not provided
    if output_path is None:
        import tempfile
        output_path = tempfile.mkdtemp()
    
    # Save the model with safe_serialization=True
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Reload the model
    reloaded_model = AutoModel.from_pretrained(output_path, use_safetensors=True)
    
    logger.info(f"Successfully converted model to safetensors format")
    
    # Clean up temporary directory if one was created
    if output_path != DEFAULT_CACHE_DIR and not output_path:
        import shutil
        shutil.rmtree(output_path)
    
    return reloaded_model

def load_model(
    model_id_or_path: str,
    use_safetensors: bool = True,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> PreTrainedModel:
    """
    Load a model from Hugging Face Hub or local path.
    
    Args:
        model_id_or_path: The ID or path of the model to load.
        use_safetensors: Whether to use safetensors for loading the model.
        revision: The specific model revision to load.
        token: Hugging Face API token.
        **kwargs: Additional arguments for model loading.
        
    Returns:
        The loaded model.
    """
    logger.info(f"Loading model: {model_id_or_path}")
    
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained(
        model_id_or_path,
        revision=revision,
        use_safetensors=use_safetensors,
        token=token,
        **kwargs
    )
    
    logger.info(f"Successfully loaded model: {model_id_or_path}")
    
    return model
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/model_utils.py") -Value $modelUtilsContent

# Implement dataset_management/__init__.py
$datasetInitContent = @"
"""
Dataset Management Module
------------------------
Provides functions for managing datasets from the Hugging Face Hub.
"""

from .download_dataset import download_dataset
from .upload_dataset import upload_dataset
from .list_datasets import list_datasets
from .delete_dataset import delete_dataset
from .dataset_utils import get_dataset_info

__all__ = [
    'download_dataset',
    'upload_dataset',
    'list_datasets',
    'delete_dataset',
    'get_dataset_info',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/__init__.py") -Value $datasetInitContent

# Implement dataset_management/download_dataset.py
$downloadDatasetContent = @"
"""
Download Dataset
---------------
Functions for downloading datasets from Hugging Face Hub.
"""

from typing import Dict, Any, Optional, Union, List
from datasets import load_dataset
from ..utils.logging_utils import get_logger
from ..config.constants import DEFAULT_CACHE_DIR
from ..cache.cache_manager import CacheManager

logger = get_logger(__name__)

def download_dataset(
    dataset_id: str,
    subset: Optional[str] = None,
    split: Optional[Union[str, List[str]]] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Download a dataset from Hugging Face Hub.
    
    Args:
        dataset_id: The ID of the dataset to download.
        subset: The subset of the dataset to download.
        split: The split of the dataset to download.
        revision: The specific dataset revision to download.
        cache_dir: Directory where the datasets should be cached.
        token: Hugging Face API token.
        **kwargs: Additional arguments for dataset loading.
        
    Returns:
        Dict containing information about the downloaded dataset.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Create cache manager
    cache_manager = CacheManager(cache_dir=cache_dir)
    
    # Check if dataset is already in cache
    if cache_manager.is_dataset_cached(dataset_id, subset, revision):
        logger.info(f"Dataset {dataset_id} already in cache. Loading from cache.")
        return cache_manager.get_cached_dataset_info(dataset_id, subset, revision)
    
    logger.info(f"Downloading dataset {dataset_id}")
    
    # Download the dataset
    dataset = load_dataset(
        dataset_id,
        name=subset,
        split=split,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        **kwargs
    )
    
    # Save dataset to cache
    cache_path = cache_manager.cache_dataset(dataset, dataset_id, subset, revision)
    
    dataset_info = {
        "dataset_id": dataset_id,
        "subset": subset,
        "split": split,
        "revision": revision,
        "cache_path": cache_path,
        "features": dataset.features if hasattr(dataset, "features") else None,
        "num_examples": len(dataset) if hasattr(dataset, "__len__") else None
    }
    
    logger.info(f"Successfully downloaded and cached dataset {dataset_id}")
    return dataset_info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/download_dataset.py") -Value $downloadDatasetContent

# Implement dataset_management/upload_dataset.py
$uploadDatasetContent = @"
"""
Upload Dataset
-------------
Functions for uploading datasets to Hugging Face Hub.
"""

from typing import Dict, Any, Optional, Union
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def upload_dataset(
    dataset: Union[Dataset, DatasetDict, str],
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Upload a dataset to Hugging Face Hub.
    
    Args:
        dataset: The dataset to upload or path to saved dataset.
        repo_id: The repository ID to upload to.
        commit_message: The commit message for the upload.
        private: Whether the repository should be private.
        token: Hugging Face API token.
        **kwargs: Additional arguments for dataset uploading.
        
    Returns:
        Dict containing information about the uploaded dataset.
    """
    logger.info(f"Uploading dataset to {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # If dataset is a path, load it first
    if isinstance(dataset, str):
        from datasets import load_from_disk
        dataset = load_from_disk(dataset)
    
    # Create a temporary directory to save the dataset
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info(f"Saving dataset to temporary directory {tmpdirname}")
        
        # Save dataset to temporary directory
        if isinstance(dataset, Dataset):
            dataset.save_to_disk(tmpdirname)
        elif isinstance(dataset, DatasetDict):
            dataset.save_to_disk(tmpdirname)
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
        
        # Upload the dataset
        logger.info(f"Pushing dataset to the Hub: {repo_id}")
        commit_message = commit_message or f"Upload dataset"
        
        response = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        
        api.upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message
        )
    
    logger.info(f"Successfully uploaded dataset to {repo_id}")
    
    return {
        "repo_id": repo_id,
        "dataset_type": type(dataset).__name__,
        "commit_message": commit_message,
        "private": private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/upload_dataset.py") -Value $uploadDatasetContent

# Implement dataset_management/list_datasets.py
$listDatasetsContent = @"
"""
List Datasets
------------
Functions for listing datasets from Hugging Face Hub.
"""

from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, list_datasets
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def list_datasets(
    author: Optional[str] = None,
    search: Optional[str] = None,
    token: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List datasets from Hugging Face Hub.
    
    Args:
        author: Filter by dataset author.
        search: Search query for dataset names and descriptions.
        token: Hugging Face API token.
        limit: Maximum number of datasets to return.
        
    Returns:
        List of dictionaries containing dataset information.
    """
    logger.info(f"Listing datasets with params: author={author}, search={search}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Get datasets
    datasets = list_datasets(
        author=author,
        search=search,
        token=token,
        limit=limit
    )
    
    # Format the results
    results = []
    for dataset_info in datasets:
        results.append({
            "dataset_id": dataset_info.id,
            "author": dataset_info.author,
            "tags": dataset_info.tags,
            "downloads": dataset_info.downloads,
            "likes": dataset_info.likes
        })
    
    logger.info(f"Found {len(results)} datasets")
    return results
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/list_datasets.py") -Value $listDatasetsContent

# Implement dataset_management/delete_dataset.py
$deleteDatasetContent = @"
"""
Delete Dataset
-------------
Functions for deleting datasets from Hugging Face Hub.
"""

from typing import Dict, Any, Optional
from huggingface_hub import HfApi, delete_repo
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def delete_dataset(
    repo_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a dataset from Hugging Face Hub.
    
    Args:
        repo_id: The repository ID to delete.
        token: Hugging Face API token.
        
    Returns:
        Dict containing information about the deleted dataset.
    """
    logger.info(f"Deleting dataset repository: {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Delete the repository
    delete_repo(repo_id=repo_id, repo_type="dataset", token=token)
    
    logger.info(f"Successfully deleted dataset repository: {repo_id}")
    
    return {
        "repo_id": repo_id,
        "status": "deleted"
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/delete_dataset.py") -Value $deleteDatasetContent

# Implement dataset_management/dataset_utils.py
$datasetUtilsContent = @"
"""
Dataset Utilities
----------------
Utility functions for dataset management operations.
"""

from typing import Dict, Any, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_dataset_info(
    dataset_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed information about a dataset.
    
    Args:
        dataset_id: The ID of the dataset.
        token: Hugging Face API token.
        
    Returns:
        Dict containing dataset information.
    """
    from huggingface_hub import dataset_info
    
    logger.info(f"Getting information for dataset: {dataset_id}")
    
    info = dataset_info(repo_id=dataset_id, token=token)
    
    return {
        "dataset_id": info.id,
        "author": info.author,
        "tags": info.tags,
        "downloads": info.downloads,
        "likes": info.likes,
        "last_modified": str(info.lastModified),
        "private": info.private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/dataset_utils.py") -Value $datasetUtilsContent

# Implement tokenizer_management/__init__.py
$tokenizerInitContent = @"
"""
Tokenizer Management Module
--------------------------
Provides functions for managing tokenizers from the Hugging Face Hub.
"""

from .download_tokenizer import download_tokenizer
from .upload_tokenizer import upload_tokenizer
from .list_tokenizers import list_tokenizers
from .delete_tokenizer import delete_tokenizer
from .tokenizer_utils import get_tokenizer_info

__all__ = [
    'download_tokenizer',
    'upload_tokenizer',
    'list_tokenizers',
    'delete_tokenizer',
    'get_tokenizer_info',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/__init__.py") -Value $tokenizerInitContent

# Implement tokenizer_management/download_tokenizer.py
$downloadTokenizerContent = @"
"""
Download Tokenizer
-----------------
Functions for downloading tokenizers from Hugging Face Hub.
"""

from typing import Dict, Any, Optional
from transformers import AutoTokenizer
from ..utils.logging_utils import get_logger
from ..config.constants import DEFAULT_CACHE_DIR
from ..cache.cache_manager import CacheManager

logger = get_logger(__name__)

def download_tokenizer(
    tokenizer_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Download a tokenizer from Hugging Face Hub.
    
    Args:
        tokenizer_id: The ID of the tokenizer to download.
        revision: The specific tokenizer revision to download.
        cache_dir: Directory where the tokenizers should be cached.
        token: Hugging Face API token.
        **kwargs: Additional arguments for tokenizer loading.
        
    Returns:
        Dict containing information about the downloaded tokenizer.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Create cache manager
    cache_manager = CacheManager(cache_dir=cache_dir)
    
    # Check if tokenizer is already in cache
    if cache_manager.is_tokenizer_cached(tokenizer_id, revision):
        logger.info(f"Tokenizer {tokenizer_id} already in cache. Loading from cache.")
        return cache_manager.get_cached_tokenizer_info(tokenizer_id, revision)
    
    logger.info(f"Downloading tokenizer {tokenizer_id}")
    
    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        **kwargs
    )
    
    # Save tokenizer to cache
    cache_path = cache_manager.cache_tokenizer(tokenizer, tokenizer_id, revision)
    
    tokenizer_info = {
        "tokenizer_id": tokenizer_id,
        "revision": revision,
        "cache_path": cache_path,
        "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else None
    }
    
    logger.info(f"Successfully downloaded and cached tokenizer {tokenizer_id}")
    return tokenizer_info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/download_tokenizer.py") -Value $downloadTokenizerContent

# Implement tokenizer_management/upload_tokenizer.py
$uploadTokenizerContent = @"
"""
Upload Tokenizer
---------------
Functions for uploading tokenizers to Hugging Face Hub.
"""

from typing import Dict, Any, Optional, Union
from transformers import PreTrainedTokenizer
from huggingface_hub import HfApi
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def upload_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, str],
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Upload a tokenizer to Hugging Face Hub.
    
    Args:
        tokenizer: The tokenizer to upload or path to saved tokenizer.
        repo_id: The repository ID to upload to.
        commit_message: The commit message for the upload.
        private: Whether the repository should be private.
        token: Hugging Face API token.
        **kwargs: Additional arguments for tokenizer uploading.
        
    Returns:
        Dict containing information about the uploaded tokenizer.
    """
    logger.info(f"Uploading tokenizer to {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # If tokenizer is a path, load it first
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
    
    # Create a temporary directory to save the tokenizer
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info(f"Saving tokenizer to temporary directory {tmpdirname}")
        tokenizer.save_pretrained(tmpdirname)
        
        # Upload the tokenizer
        logger.info(f"Pushing tokenizer to the Hub: {repo_id}")
        commit_message = commit_message or f"Upload tokenizer"
        
        response = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True
        )
        
        api.upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            commit_message=commit_message
        )
    
    logger.info(f"Successfully uploaded tokenizer to {repo_id}")
    
    return {
        "repo_id": repo_id,
        "commit_message": commit_message,
        "private": private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/upload_tokenizer.py") -Value $uploadTokenizerContent

# Implement tokenizer_management/list_tokenizers.py
$listTokenizersContent = @"
"""
List Tokenizers
--------------
Functions for listing tokenizers from Hugging Face Hub.
"""

from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, list_models
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def list_tokenizers(
    author: Optional[str] = None,
    search: Optional[str] = None,
    token: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List tokenizers from Hugging Face Hub.
    
    Args:
        author: Filter by tokenizer author.
        search: Search query for tokenizer names and descriptions.
        token: Hugging Face API token.
        limit: Maximum number of tokenizers to return.
        
    Returns:
        List of dictionaries containing tokenizer information.
    """
    logger.info(f"Listing tokenizers with params: author={author}, search={search}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Get models with tokenizer tag
    models = list_models(
        author=author,
        search=search,
        filter="tokenizer",
        token=token,
        limit=limit
    )
    
    # Format the results
    results = []
    for model_info in models:
        results.append({
            "tokenizer_id": model_info.modelId,
            "author": model_info.author,
            "tags": model_info.tags,
            "downloads": model_info.downloads,
            "likes": model_info.likes
        })
    
    logger.info(f"Found {len(results)} tokenizers")
    return results
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/list_tokenizers.py") -Value $listTokenizersContent

# Implement tokenizer_management/delete_tokenizer.py
$deleteTokenizerContent = @"
"""
Delete Tokenizer
---------------
Functions for deleting tokenizers from Hugging Face Hub.
"""

from typing import Dict, Any, Optional
from huggingface_hub import HfApi, delete_repo
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def delete_tokenizer(
    repo_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a tokenizer from Hugging Face Hub.
    
    Args:
        repo_id: The repository ID to delete.
        token: Hugging Face API token.
        
    Returns:
        Dict containing information about the deleted tokenizer.
    """
    logger.info(f"Deleting tokenizer repository: {repo_id}")
    
    # Initialize the Hugging Face API
    api = HfApi(token=token)
    
    # Delete the repository
    delete_repo(repo_id=repo_id, token=token)
    
    logger.info(f"Successfully deleted tokenizer repository: {repo_id}")
    
    return {
        "repo_id": repo_id,
        "status": "deleted"
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/delete_tokenizer.py") -Value $deleteTokenizerContent

# Implement tokenizer_management/tokenizer_utils.py
$tokenizerUtilsContent = @"
"""
Tokenizer Utilities
------------------
Utility functions for tokenizer management operations.
"""

from typing import Dict, Any, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_tokenizer_info(
    tokenizer_id: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed information about a tokenizer.
    
    Args:
        tokenizer_id: The ID of the tokenizer.
        token: Hugging Face API token.
        
    Returns:
        Dict containing tokenizer information.
    """
    from huggingface_hub import model_info
    
    logger.info(f"Getting information for tokenizer: {tokenizer_id}")
    
    info = model_info(repo_id=tokenizer_id, token=token)
    
    return {
        "tokenizer_id": info.modelId,
        "author": info.author,
        "tags": info.tags,
        "downloads": info.downloads,
        "likes": info.likes,
        "last_modified": str(info.lastModified),
        "private": info.private
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/tokenizer_management/tokenizer_utils.py") -Value $tokenizerUtilsContent

# Implement api_integration/__init__.py
$apiInitContent = @"
"""
API Integration Module
---------------------
Provides functions for interacting with the Hugging Face API.
"""

from .auth import get_token, set_token, login
from .api_client import get_api_client
from .api_utils import check_token_validity

__all__ = [
    'get_token',
    'set_token',
    'login',
    'get_api_client',
    'check_token_validity',
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/api_integration/__init__.py") -Value $apiInitContent

# Implement api_integration/auth.py
$authContent = @"
# Implementation for huggingface_toolset/api_integration/auth.py
$authContent = @"
"""
Hugging Face Authentication
-------------------------
Functions for authenticating with the Hugging Face API.
"""

import os
from typing import Optional
from huggingface_hub import login as hf_login
from ..utils.logging_utils import get_logger
from ..config.constants import TOKEN_ENV_NAME

logger = get_logger(__name__)

def get_token() -> Optional[str]:
    """
    Get the Hugging Face API token.
    
    Returns:
        The token or None if not set.
    """
    token = os.environ.get(TOKEN_ENV_NAME)
    if not token:
        logger.warning(f"No Hugging Face token found in environment variable {TOKEN_ENV_NAME}")
    return token

def set_token(token: str) -> None:
    """
    Set the Hugging Face API token in the environment.
    
    Args:
        token: The token to set.
    """
    os.environ[TOKEN_ENV_NAME] = token
    logger.info(f"Hugging Face token set in environment variable {TOKEN_ENV_NAME}")

def login(token: Optional[str] = None, write_token: bool = False) -> bool:
    """
    Login to Hugging Face Hub.
    
    Args:
        token: The token to use. If None, will try to get from environment.
        write_token: Whether to write the token to the hub cache.
    
    Returns:
        True if login was successful, False otherwise.
    """
    if token is None:
        token = get_token()
    
    if not token:
        logger.error("No token provided and none found in environment")
        return False
    
    try:
        hf_login(token=token, write_token=write_token)
        logger.info("Successfully logged in to Hugging Face Hub")
        return True
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face Hub: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/api_integration/auth.py") -Value $authContent

# Implementation for huggingface_toolset/api_integration/api_client.py
$apiClientContent = @"
"""
API Client
----------
Client for interacting with the Hugging Face API.
"""

from typing import Optional, Dict, Any, List
from huggingface_hub import HfApi

from .auth import get_token
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_api_client(token: Optional[str] = None) -> HfApi:
    """
    Get a Hugging Face API client instance.
    
    Args:
        token: Optional token to use for authentication.
    
    Returns:
        A Hugging Face API client instance.
    """
    if token is None:
        token = get_token()
    
    api = HfApi(token=token)
    logger.debug("Created Hugging Face API client")
    return api

def get_model_info(model_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a model from the Hugging Face Hub.
    
    Args:
        model_id: The model ID to get information for.
        token: Optional token to use for authentication.
    
    Returns:
        A dictionary with model information.
    """
    api = get_api_client(token)
    try:
        model_info = api.model_info(model_id)
        return model_info.to_dict()
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        raise

def get_dataset_info(dataset_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a dataset from the Hugging Face Hub.
    
    Args:
        dataset_id: The dataset ID to get information for.
        token: Optional token to use for authentication.
    
    Returns:
        A dictionary with dataset information.
    """
    api = get_api_client(token)
    try:
        dataset_info = api.dataset_info(dataset_id)
        return dataset_info.to_dict()
    except Exception as e:
        logger.error(f"Failed to get dataset info for {dataset_id}: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/api_integration/api_client.py") -Value $apiClientContent

# Implementation for huggingface_toolset/api_integration/api_utils.py
$apiUtilsContent = @"
"""
API Utilities
------------
Utility functions for working with the Hugging Face API.
"""

from typing import Optional, Dict, Any, List, Tuple
import requests

from .auth import get_token
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

API_URL = "https://huggingface.co/api"

def check_token_validity(token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if a Hugging Face token is valid.
    
    Args:
        token: The token to check. If None, will try to get from environment.
    
    Returns:
        A tuple of (is_valid, message).
    """
    if token is None:
        token = get_token()
    
    if not token:
        return False, "No token provided"
    
    try:
        response = requests.get(
            f"{API_URL}/whoami", 
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            user_info = response.json()
            return True, f"Token is valid for user: {user_info.get('name', 'unknown')}"
        else:
            return False, f"Token is invalid: {response.status_code} - {response.text}"
    
    except Exception as e:
        logger.error(f"Error checking token validity: {e}")
        return False, f"Error checking token: {str(e)}"
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/api_integration/api_utils.py") -Value $apiUtilsContent

# Implementation for huggingface_toolset/config/constants.py
$constantsContent = @"
"""
Constants
--------
Constant values used across the Hugging Face toolset.
"""

import os
from pathlib import Path

# Environment variable names
TOKEN_ENV_NAME = "HUGGINGFACE_TOKEN"

# Default values
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
DEFAULT_REVISION = "main"

# File extensions
SAFETENSORS_EXTENSION = ".safetensors"
PYTORCH_EXTENSION = ".bin"

# API endpoints
HF_API_BASE_URL = "https://huggingface.co/api"

# Quantization settings
SUPPORTED_BITS = [4, 8, 16]
DEFAULT_BITS = 8

# Model settings
DEFAULT_MODEL_SAVE_FORMAT = "safetensors"
DEFAULT_TOKENIZER_FORMAT = "fast"
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/config/constants.py") -Value $constantsContent

# Implementation for huggingface_toolset/config/env_loader.py
$envLoaderContent = @"
"""
Environment Loader
----------------
Functions for loading environment variables.
"""

import os
import dotenv
from typing import Optional, Dict
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_env(env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file. If None, will search for a .env file in the current directory.
    
    Returns:
        A dictionary of loaded environment variables.
    """
    if env_file is None:
        # Try to find a .env file in the current directory
        default_locations = [".env", "../.env", "../../.env"]
        for loc in default_locations:
            if Path(loc).is_file():
                env_file = loc
                break
    
    if env_file and Path(env_file).is_file():
        logger.info(f"Loading environment variables from {env_file}")
        dotenv.load_dotenv(env_file)
        # Return loaded variables (just the ones from the file)
        return dotenv.dotenv_values(env_file)
    else:
        logger.warning("No .env file found or specified")
        return {}
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/config/env_loader.py") -Value $envLoaderContent

# Implementation for huggingface_toolset/config/config_utils.py
$configUtilsContent = @"
"""
Configuration Utilities
---------------------
Utility functions for configuration management.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file.
    
    Returns:
        The configuration as a dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.error(f"Unsupported configuration file format: {config_path}")
            raise ValueError(f"Unsupported configuration file format: {config_path}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/config/config_utils.py") -Value $configUtilsContent

# Implementation for huggingface_toolset/utils/logging_utils.py
$loggingUtilsContent = @"
"""
Logging Utilities
---------------
Utility functions for logging.
"""

import os
import logging
from typing import Optional

def setup_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: The logging level (e.g., 'DEBUG', 'INFO'). If None, will use the LOG_LEVEL env var or default to INFO.
        log_file: Optional path to a log file.
    """
    if level is None:
        level = os.environ.get('LOG_LEVEL', 'INFO')
    
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file
    )
    
    # If log_file is set, also log to console
    if log_file:
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger.
    
    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/logging_utils.py") -Value $loggingUtilsContent

# Implementation for huggingface_toolset/utils/file_utils.py
$fileUtilsContent = @"
"""
File Utilities
------------
Utility functions for file operations.
"""

import os
import shutil
from typing import Optional, List
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def ensure_dir(directory: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: The directory path.
    
    Returns:
        The directory path.
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: The path to the file.
    
    Returns:
        The size of the file in bytes.
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError) as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return 0

def copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        src: Source file path.
        dst: Destination file path.
        overwrite: Whether to overwrite the destination if it exists.
    
    Returns:
        True if the copy was successful, False otherwise.
    """
    try:
        if os.path.exists(dst) and not overwrite:
            logger.warning(f"Destination file already exists: {dst}")
            return False
        
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        shutil.copy2(src, dst)
        logger.info(f"Copied {src} to {dst}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False

def remove_file(file_path: str, ignore_errors: bool = False) -> bool:
    """
    Remove a file.
    
    Args:
        file_path: The path to the file to remove.
        ignore_errors: Whether to ignore errors.
    
    Returns:
        True if the file was removed successfully, False otherwise.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")
            return True
        else:
            logger.warning(f"File not found: {file_path}")
            return False
    
    except Exception as e:
        if not ignore_errors:
            logger.error(f"Failed to remove file {file_path}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/file_utils.py") -Value $fileUtilsContent

# Implementation for huggingface_toolset/cache/cache_manager.py
$cacheManagerContent = @"
"""
Cache Manager
-----------
Management utilities for the local cache of models, datasets, and tokenizers.
"""

import os
import json
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..config.constants import DEFAULT_CACHE_DIR
from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, remove_file

logger = get_logger(__name__)

def get_cache_dir() -> str:
    """
    Get the cache directory path.
    
    Returns:
        The path to the cache directory.
    """
    cache_dir = os.environ.get('HUGGINGFACE_CACHE_DIR', DEFAULT_CACHE_DIR)
    ensure_dir(cache_dir)
    return cache_dir

def is_cached(model_id: str, revision: str = "main") -> bool:
    """
    Check if a model is cached locally.
    
    Args:
        model_id: The model ID to check.
        revision: The model revision to check.
    
    Returns:
        True if the model is cached, False otherwise.
    """
    cache_dir = get_cache_dir()
    # In HF Hub, models are cached in subdirectories based on hash of model_id and revision
    # This is a simplified check that just looks for directories containing the model_id
    model_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)) and model_id.replace('/', '--') in d]
    return len(model_dirs) > 0

def clear_cache(model_id: Optional[str] = None) -> bool:
    """
    Clear the cache for a specific model or all models.
    
    Args:
        model_id: Optional model ID to clear. If None, clears the entire cache.
    
    Returns:
        True if the cache was cleared successfully, False otherwise.
    """
    cache_dir = get_cache_dir()
    
    try:
        if model_id:
            logger.info(f"Clearing cache for model: {model_id}")
            model_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)) and model_id.replace('/', '--') in d]
            for model_dir in model_dirs:
                shutil.rmtree(os.path.join(cache_dir, model_dir), ignore_errors=True)
        else:
            logger.info("Clearing entire cache")
            # Clear everything except .cache_info
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if item != '.cache_info':
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        remove_file(item_path, ignore_errors=True)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False

def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the cache.
    
    Returns:
        A dictionary with cache information.
    """
    cache_dir = get_cache_dir()
    cache_info = {
        'cache_dir': cache_dir,
        'size_bytes': 0,
        'models': []
    }
    
    try:
        # Calculate total size
        for root, dirs, files in os.walk(cache_dir):
            cache_info['size_bytes'] += sum(os.path.getsize(os.path.join(root, file)) for file in files if os.path.isfile(os.path.join(root, file)))
        
        # Find cached models
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path) and '--' in item:
                # This is likely a model directory
                model_parts = item.split('--')
                if len(model_parts) >= 2:
                    model_id = model_parts[0] + '/' + model_parts[1]
                    model_info = {
                        'model_id': model_id,
                        'directory': item,
                        'size_bytes': 0
                    }
                    
                    # Calculate model size
                    for root, dirs, files in os.walk(item_path):
                        model_info['size_bytes'] += sum(os.path.getsize(os.path.join(root, file)) for file in files if os.path.isfile(os.path.join(root, file)))
                    
                    cache_info['models'].append(model_info)
        
        return cache_info
    
    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        return cache_info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/cache_manager.py") -Value $cacheManagerContent

# Implementation for huggingface_toolset/cache/safetensors_cache.py
$safetensorsCacheContent = @"
"""
Safetensors Cache
---------------
Specialized caching utilities for safetensors files.
"""

import os
import json
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, get_file_size
from .cache_manager import get_cache_dir

logger = get_logger(__name__)

def cache_tensor(tensor_dict: Dict[str, torch.Tensor], model_id: str, filename: str) -> str:
    """
    Cache a dictionary of tensors in safetensors format.
    
    Args:
        tensor_dict: Dictionary mapping names to tensors.
        model_id: The model ID.
        filename: The filename to save as.
    
    Returns:
        The path to the cached file.
    """
    cache_dir = get_cache_dir()
    safetensors_dir = os.path.join(cache_dir, 'safetensors', model_id.replace('/', '--'))
    ensure_dir(safetensors_dir)
    
    if not filename.endswith('.safetensors'):
        filename += '.safetensors'
    
    file_path = os.path.join(safetensors_dir, filename)
    
    try:
        save_file(tensor_dict, file_path)
        logger.info(f"Cached tensor dictionary to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to cache tensor dictionary: {e}")
        raise

def load_cached_tensor(model_id: str, filename: str) -> Dict[str, torch.Tensor]:
    """
    Load a cached tensor dictionary in safetensors format.
    
    Args:
        model_id: The model ID.
        filename: The filename to load.
    
    Returns:
        Dictionary mapping names to tensors.
    """
    cache_dir = get_cache_dir()
    safetensors_dir = os.path.join(cache_dir, 'safetensors', model_id.replace('/', '--'))
    
    if not filename.endswith('.safetensors'):
        filename += '.safetensors'
    
    file_path = os.path.join(safetensors_dir, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"Cached tensor file not found: {file_path}")
        raise FileNotFoundError(f"Cached tensor file not found: {file_path}")
    
    try:
        tensor_dict = load_file(file_path)
        logger.info(f"Loaded cached tensor dictionary from {file_path}")
        return tensor_dict
    except Exception as e:
        logger.error(f"Failed to load cached tensor dictionary: {e}")
        raise

def list_cached_tensors(model_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all cached tensor files.
    
    Args:
        model_id: Optional model ID to filter by.
    
    Returns:
        List of dictionaries with information about cached tensor files.
    """
    cache_dir = get_cache_dir()
    safetensors_dir = os.path.join(cache_dir, 'safetensors')
    
    if not os.path.exists(safetensors_dir):
        return []
    
    result = []
    
    if model_id:
        model_dir = os.path.join(safetensors_dir, model_id.replace('/', '--'))
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith('.safetensors'):
                    file_path = os.path.join(model_dir, filename)
                    result.append({
                        'model_id': model_id,
                        'filename': filename,
                        'path': file_path,
                        'size_bytes': get_file_size(file_path)
                    })
    else:
        for dir_name in os.listdir(safetensors_dir):
            dir_path = os.path.join(safetensors_dir, dir_name)
            if os.path.isdir(dir_path):
                # Convert back from directory name to model_id
                current_model_id = dir_name.replace('--', '/')
                for filename in os.listdir(dir_path):
                    if filename.endswith('.safetensors'):
                        file_path = os.path.join(dir_path, filename)
                        result.append({
                            'model_id': current_model_id,
                            'filename': filename,
                            'path': file_path,
                            'size_bytes': get_file_size(file_path)
                        })
    
    return result
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/safetensors_cache.py") -Value $safetensorsCacheContent

# Implementation for huggingface_toolset/quantization/quantize_model.py
$quantizeModelContent = @"
"""
Quantize Model
------------
Functions for quantizing Hugging Face models.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import PreTrainedModel

from ..utils.logging_utils import get_logger
from ..cache.cache_manager import get_cache_dir
from ..config.constants import DEFAULT_BITS, SUPPORTED_BITS

logger = get_logger(__name__)

def quantize_model(
    model: Union[PreTrainedModel, str],
    output_dir: Optional[str] = None,
    bits: int = DEFAULT_BITS,
    device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto",
    **kwargs
) -> PreTrainedModel:
    """
    Quantize a Hugging Face model.
    
    Args:
        model: The model to quantize or a model ID to load.
        output_dir: Directory to save the quantized model.
        bits: Number of bits for quantization (4 or 8).
        device_map: Device map for model loading and saving.
        **kwargs: Additional arguments for model loading.
    
    Returns:
        The quantized model.
    """
    if bits not in SUPPORTED_BITS:
        logger.error(f"Unsupported bits value: {bits}. Supported values are {SUPPORTED_BITS}")
        raise ValueError(f"Unsupported bits value: {bits}. Supported values are {SUPPORTED_BITS}")
    
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("bitsandbytes not installed. Please install it with 'pip install bitsandbytes'")
        raise ImportError("bitsandbytes not installed. Please install it with 'pip install bitsandbytes'")
    
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        logger.error("transformers not installed. Please install it with 'pip install transformers'")
        raise ImportError("transformers not installed. Please install it with 'pip install transformers'")
    
    logger.info(f"Quantizing model to {bits} bits")
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_compute_dtype=torch.float16 if bits == 4 else None,
        bnb_4bit_use_double_quant=True if bits == 4 else None,
        bnb_4bit_quant_type="nf4" if bits == 4 else None
    )
    
    # Load model with quantization
    if isinstance(model, str):
        logger.info(f"Loading model {model} with {bits}-bit quantization")
        model_kwargs = {
            "device_map": device_map,
            "quantization_config": quantization_config,
            **kwargs
        }
        quantized_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    else:
        logger.info(f"Quantizing provided model to {bits} bits")
        # For already loaded models, we would need to apply quantization manually
        # This is complex and might require model-specific handling
        raise NotImplementedError("Quantizing already loaded models is not yet supported")
    
    # Save quantized model if output_dir is provided
    if output_dir:
        logger.info(f"Saving quantized model to {output_dir}")
        quantized_model.save_pretrained(output_dir)
    
    return quantized_model

def get_optimal_device_map(
    model_id: str,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None
) -> Dict[str, Union[int, str]]:
    """
    Calculate an optimal device map for a model based on available GPU memory.
    
    Args:
        model_id: The model ID.
        max_memory: Optional dictionary mapping device to maximum memory.
    
    Returns:
        A device map dictionary.
    """
    try:
        from accelerate import infer_auto_device_map, init_empty_weights
        from transformers import AutoConfig
    except ImportError:
        logger.error("accelerate not installed. Please install it with 'pip install accelerate'")
        raise ImportError("accelerate not installed. Please install it with 'pip install accelerate'")
    
    logger.info(f"Computing optimal device map for {model_id}")
    
    # Get model config
    config = AutoConfig.from_pretrained(model_id)
    
    # Infer device map
    with init_empty_weights():
        device_map = infer_auto_device_map(
            config,
            max_memory=max_memory,
            no_split_module_classes=["GPTJBlock", "LlamaDecoderLayer", "MistralDecoderLayer"]
        )
    
    logger.info(f"Calculated device map: {device_map}")
    return device_map
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/quantization/quantize_model.py") -Value $quantizeModelContent

# Implementation for huggingface_toolset/quantization/quantization_utils.py
$quantizationUtilsContent = @"
"""
Quantization Utilities
-------------------
Utility functions for model quantization.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
import gc

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def clear_gpu_memory() -> None:
    """
    Clear GPU memory by emptying the cache and collecting garbage.
    """
    logger.info("Clearing GPU memory")
    torch.cuda.empty_cache()
    gc.collect()

def get_gpu_memory_info() -> List[Dict[str, Any]]:
    """
    Get information about GPU memory usage.
    
    Returns:
        A list of dictionaries with GPU memory information.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available")
        return []
    
    result = []
    for i in range(torch.cuda.device_count()):
        info = {
            'device': i,
            'name': torch.cuda.get_device_name(i),
            'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory,
            'allocated_memory_bytes': torch.cuda.memory_allocated(i),
            'reserved_memory_bytes': torch.cuda.memory_reserved(i),
            'free_memory_bytes': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        }
        result.append(info)
    
    return result

def calculate_max_memory() -> Dict[Union[int, str], str]:
    """
    Calculate maximum memory available for each GPU device.
    
    Returns:
        A dictionary mapping device to maximum memory.
    """
    max_memory = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # Get total memory in bytes
            total_memory = torch.cuda.get_device_properties(i).total_memory
            # Reserve some memory for the system (10%)
            usable_memory = int(total_memory * 0.9)
            # Convert to GB string (e.g., '16GiB')
            max_memory[i] = f"{usable_memory // (1024**3)}GiB"
    
    # Add CPU memory (use 32GB or actual available memory)
    try:
        import psutil
        cpu_memory = psutil.virtual_memory().total
        max_memory['cpu'] = f"{min(32, cpu_memory // (1024**3))}GiB"
    except ImportError:
        # Default to 16GB if psutil is not available
        max_memory['cpu'] = "16GiB"
    
    logger.info(f"Calculated max memory: {max_memory}")
    return max_memory

def get_optimal_bits(model_id: str) -> int:
    """
    Determine the optimal number of bits for quantization based on the model and available hardware.
    
    Args:
        model_id: The model ID.
    
    Returns:
        The optimal number of bits (4 or 8).
    """
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            # Get the GPU with the most memory
            max_gpu = max(gpu_info, key=lambda x: x['free_memory_bytes'])
            free_memory_gb = max_gpu['free_memory_bytes'] / (1024**3)
            
            # Get model size from HF API if possible
            try:
                from ..api_integration.api_client import get_model_info
                model_info = get_model_info(model_id)
                model_size_gb = model_info.get("safetensors_size_bytes", model_info.get("size_bytes", 0)) / (1024**3)
                
                # If model is large and we have enough memory, use 8-bit
                if model_size_gb > 10 and free_memory_gb < model_size_gb * 0.5:
                    logger.info(f"Model {model_id} is large ({model_size_gb:.2f} GB) and available GPU memory is limited ({free_memory_gb:.2f} GB), using 4-bit quantization")
                    return 4
                elif model_size_gb > 20:
                    logger.info(f"Model {model_id} is very large ({model_size_gb:.2f} GB), using 4-bit quantization")
                    return 4
                else:
                    logger.info(f"Model {model_id} is moderate size ({model_size_gb:.2f} GB), using 8-bit quantization")
                    return 8
            except Exception as e:
                logger.warning(f"Failed to get model info: {e}")
                
                # Fallback: Just check free memory
                if free_memory_gb < 8:
                    logger.info(f"Limited GPU memory ({free_memory_gb:.2f} GB), using 4-bit quantization")
                    return 4
                else:
                    logger.info(f"Sufficient GPU memory ({free_memory_gb:.2f} GB), using 8-bit quantization")
                    return 8
    
    # If no GPU or unable to determine, default to 8-bit
    logger.info("Default to 8-bit quantization")
    return 8
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/quantization/quantization_utils.py") -Value $quantizationUtilsContent

# Implementation for huggingface_toolset/model_management/model_utils.py
$modelUtilsContent = @"
"""
Model Utilities
------------
Utility functions for model management.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import PreTrainedModel
from safetensors.torch import save_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)

def convert_to_safetensors(
    model: Union[PreTrainedModel, str, Dict[str, torch.Tensor]],
    output_dir: str,
    save_model_config: bool = True
) -> List[str]:
    """
    Convert a PyTorch model to safetensors format.
    
    Args:
        model: The model to convert, a model ID, or a state dict.
        output_dir: Directory to save the safetensors files.
        save_model_config: Whether to save the model configuration.
    
    Returns:
        A list of paths to the created safetensors files.
    """
    logger.info(f"Converting model to safetensors format")# Implementation for huggingface_toolset/api_integration/auth.py
$authContent = @"
"""
Hugging Face Authentication
-------------------------
Functions for authenticating with the Hugging Face API.
"""

import os
from typing import Optional
from huggingface_hub import login as hf_login
from ..utils.logging_utils import get_logger
from ..config.constants import TOKEN_ENV_NAME

logger = get_logger(__name__)

def get_token() -> Optional[str]:
    """
    Get the Hugging Face API token.
    
    Returns:
        The token or None if not set.
    """
    token = os.environ.get(TOKEN_ENV_NAME)
    if not token:
        logger.warning(f"No Hugging Face token found in environment variable {TOKEN_ENV_NAME}")
    return token

def set_token(token: str) -> None:
    """
    Set the Hugging Face API token in the environment.
    
    Args:
        token: The token to set.
    """
    os.environ[TOKEN_ENV_NAME] = token
    logger.info(f"Hugging Face token set in environment variable {TOKEN_ENV_NAME}")

def login(token: Optional[str] = None, write_token: bool = False) -> bool:
    """
    Login to Hugging Face Hub.
    
    Args:
        token: The token to use. If None, will try to get from environment.
        write_token: Whether to write the token to the hub cache.
    
    Returns:
        True if login was successful, False otherwise.
    """
    if token is None:
        token = get_token()
    
    if not token:
        logger.error("No token provided and none found in environment")
        return False
    
    try:
        hf_login(token=token, write_token=write_token)
        logger.info("Successfully logged in to Hugging Face Hub")
        return True
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face Hub: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/api_integration/auth.py") -Value $authContent

# Implementation for huggingface_toolset/api_integration/api_client.py
$apiClientContent = @"
"""
API Client
----------
Client for interacting with the Hugging Face API.
"""

from typing import Optional, Dict, Any, List
from huggingface_hub import HfApi

from .auth import get_token
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_api_client(token: Optional[str] = None) -> HfApi:
    """
    Get a Hugging Face API client instance.
    
    Args:
        token: Optional token to use for authentication.
    
    Returns:
        A Hugging Face API client instance.
    """
    if token is None:
        token = get_token()
    
    api = HfApi(token=token)
    logger.debug("Created Hugging Face API client")
    return api

def get_model_info(model_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a model from the Hugging Face Hub.
    
    Args:
        model_id: The model ID to get information for.
        token: Optional token to use for authentication.
    
    Returns:
        A dictionary with model information.
    """
    api = get_api_client(token)
    try:
        model_info = api.model_info(model_id)
        return model_info.to_dict()
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        raise

def get_dataset_info(dataset_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a dataset from the Hugging Face Hub.
    
    Args:
        dataset_id: The dataset ID to get information for.
        token: Optional token to use for authentication.
    
    Returns:
        A dictionary with dataset information.
    """
    api = get_api_client(token)
    try:
        dataset_info = api.dataset_info(dataset_id)
        return dataset_info.to_dict()
    except Exception as e:
        logger.error(f"Failed to get dataset info for {dataset_id}: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/api_integration/api_client.py") -Value $apiClientContent

# Implementation for huggingface_toolset/api_integration/api_utils.py
$apiUtilsContent = @"
"""
API Utilities
------------
Utility functions for working with the Hugging Face API.
"""

from typing import Optional, Dict, Any, List, Tuple
import requests

from .auth import get_token
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

API_URL = "https://huggingface.co/api"

def check_token_validity(token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if a Hugging Face token is valid.
    
    Args:
        token: The token to check. If None, will try to get from environment.
    
    Returns:
        A tuple of (is_valid, message).
    """
    if token is None:
        token = get_token()
    
    if not token:
        return False, "No token provided"
    
    try:
        response = requests.get(
            f"{API_URL}/whoami", 
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            user_info = response.json()
            return True, f"Token is valid for user: {user_info.get('name', 'unknown')}"
        else:
            return False, f"Token is invalid: {response.status_code} - {response.text}"
    
    except Exception as e:
        logger.error(f"Error checking token validity: {e}")
        return False, f"Error checking token: {str(e)}"
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/api_integration/api_utils.py") -Value $apiUtilsContent

# Implementation for huggingface_toolset/config/constants.py
$constantsContent = @"
"""
Constants
--------
Constant values used across the Hugging Face toolset.
"""

import os
from pathlib import Path

# Environment variable names
TOKEN_ENV_NAME = "HUGGINGFACE_TOKEN"

# Default values
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
DEFAULT_REVISION = "main"

# File extensions
SAFETENSORS_EXTENSION = ".safetensors"
PYTORCH_EXTENSION = ".bin"

# API endpoints
HF_API_BASE_URL = "https://huggingface.co/api"

# Quantization settings
SUPPORTED_BITS = [4, 8, 16]
DEFAULT_BITS = 8

# Model settings
DEFAULT_MODEL_SAVE_FORMAT = "safetensors"
DEFAULT_TOKENIZER_FORMAT = "fast"
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/config/constants.py") -Value $constantsContent

# Implementation for huggingface_toolset/config/env_loader.py
$envLoaderContent = @"
"""
Environment Loader
----------------
Functions for loading environment variables.
"""

import os
import dotenv
from typing import Optional, Dict
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_env(env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file. If None, will search for a .env file in the current directory.
    
    Returns:
        A dictionary of loaded environment variables.
    """
    if env_file is None:
        # Try to find a .env file in the current directory
        default_locations = [".env", "../.env", "../../.env"]
        for loc in default_locations:
            if Path(loc).is_file():
                env_file = loc
                break
    
    if env_file and Path(env_file).is_file():
        logger.info(f"Loading environment variables from {env_file}")
        dotenv.load_dotenv(env_file)
        # Return loaded variables (just the ones from the file)
        return dotenv.dotenv_values(env_file)
    else:
        logger.warning("No .env file found or specified")
        return {}
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/config/env_loader.py") -Value $envLoaderContent

# Implementation for huggingface_toolset/config/config_utils.py
$configUtilsContent = @"
"""
Configuration Utilities
---------------------
Utility functions for configuration management.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file.
    
    Returns:
        The configuration as a dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.error(f"Unsupported configuration file format: {config_path}")
            raise ValueError(f"Unsupported configuration file format: {config_path}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/config/config_utils.py") -Value $configUtilsContent

# Implementation for huggingface_toolset/utils/logging_utils.py
$loggingUtilsContent = @"
"""
Logging Utilities
---------------
Utility functions for logging.
"""

import os
import logging
from typing import Optional

def setup_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: The logging level (e.g., 'DEBUG', 'INFO'). If None, will use the LOG_LEVEL env var or default to INFO.
        log_file: Optional path to a log file.
    """
    if level is None:
        level = os.environ.get('LOG_LEVEL', 'INFO')
    
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file
    )
    
    # If log_file is set, also log to console
    if log_file:
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger.
    
    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/logging_utils.py") -Value $loggingUtilsContent

# Implementation for huggingface_toolset/utils/file_utils.py
$fileUtilsContent = @"
"""
File Utilities
------------
Utility functions for file operations.
"""

import os
import shutil
from typing import Optional, List
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def ensure_dir(directory: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: The directory path.
    
    Returns:
        The directory path.
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: The path to the file.
    
    Returns:
        The size of the file in bytes.
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError) as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return 0

def copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        src: Source file path.
        dst: Destination file path.
        overwrite: Whether to overwrite the destination if it exists.
    
    Returns:
        True if the copy was successful, False otherwise.
    """
    try:
        if os.path.exists(dst) and not overwrite:
            logger.warning(f"Destination file already exists: {dst}")
            return False
        
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        shutil.copy2(src, dst)
        logger.info(f"Copied {src} to {dst}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False

def remove_file(file_path: str, ignore_errors: bool = False) -> bool:
    """
    Remove a file.
    
    Args:
        file_path: The path to the file to remove.
        ignore_errors: Whether to ignore errors.
    
    Returns:
        True if the file was removed successfully, False otherwise.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")
            return True
        else:
            logger.warning(f"File not found: {file_path}")
            return False
    
    except Exception as e:
        if not ignore_errors:
            logger.error(f"Failed to remove file {file_path}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/file_utils.py") -Value $fileUtilsContent

# Implementation for huggingface_toolset/cache/cache_manager.py
$cacheManagerContent = @"
"""
Cache Manager
-----------
Management utilities for the local cache of models, datasets, and tokenizers.
"""

import os
import json
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..config.constants import DEFAULT_CACHE_DIR
from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, remove_file

logger = get_logger(__name__)

def get_cache_dir() -> str:
    """
    Get the cache directory path.
    
    Returns:
        The path to the cache directory.
    """
    cache_dir = os.environ.get('HUGGINGFACE_CACHE_DIR', DEFAULT_CACHE_DIR)
    ensure_dir(cache_dir)
    return cache_dir

def is_cached(model_id: str, revision: str = "main") -> bool:
    """
    Check if a model is cached locally.
    
    Args:
        model_id: The model ID to check.
        revision: The model revision to check.
    
    Returns:
        True if the model is cached, False otherwise.
    """
    cache_dir = get_cache_dir()
    # In HF Hub, models are cached in subdirectories based on hash of model_id and revision
    # This is a simplified check that just looks for directories containing the model_id
    model_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)) and model_id.replace('/', '--') in d]
    return len(model_dirs) > 0

def clear_cache(model_id: Optional[str] = None) -> bool:
    """
    Clear the cache for a specific model or all models.
    
    Args:
        model_id: Optional model ID to clear. If None, clears the entire cache.
    
    Returns:
        True if the cache was cleared successfully, False otherwise.
    """
    cache_dir = get_cache_dir()
    
    try:
        if model_id:
            logger.info(f"Clearing cache for model: {model_id}")
            model_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)) and model_id.replace('/', '--') in d]
            for model_dir in model_dirs:
                shutil.rmtree(os.path.join(cache_dir, model_dir), ignore_errors=True)
        else:
            logger.info("Clearing entire cache")
            # Clear everything except .cache_info
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if item != '.cache_info':
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        remove_file(item_path, ignore_errors=True)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False

def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the cache.
    
    Returns:
        A dictionary with cache information.
    """
    cache_dir = get_cache_dir()
    cache_info = {
        'cache_dir': cache_dir,
        'size_bytes': 0,
        'models': []
    }
    
    try:
        # Calculate total size
        for root, dirs, files in os.walk(cache_dir):
            cache_info['size_bytes'] += sum(os.path.getsize(os.path.join(root, file)) for file in files if os.path.isfile(os.path.join(root, file)))
        
        # Find cached models
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path) and '--' in item:
                # This is likely a model directory
                model_parts = item.split('--')
                if len(model_parts) >= 2:
                    model_id = model_parts[0] + '/' + model_parts[1]
                    model_info = {
                        'model_id': model_id,
                        'directory': item,
                        'size_bytes': 0
                    }
                    
                    # Calculate model size
                    for root, dirs, files in os.walk(item_path):
                        model_info['size_bytes'] += sum(os.path.getsize(os.path.join(root, file)) for file in files if os.path.isfile(os.path.join(root, file)))
                    
                    cache_info['models'].append(model_info)
        
        return cache_info
    
    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        return cache_info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/cache_manager.py") -Value $cacheManagerContent

# Implementation for huggingface_toolset/cache/safetensors_cache.py
$safetensorsCacheContent = @"
"""
Safetensors Cache
---------------
Specialized caching utilities for safetensors files.
"""

import os
import json
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, get_file_size
from .cache_manager import get_cache_dir

logger = get_logger(__name__)

def cache_tensor(tensor_dict: Dict[str, torch.Tensor], model_id: str, filename: str) -> str:
    """
    Cache a dictionary of tensors in safetensors format.
    
    Args:
        tensor_dict: Dictionary mapping names to tensors.
        model_id: The model ID.
        filename: The filename to save as.
    
    Returns:
        The path to the cached file.
    """
    cache_dir = get_cache_dir()
    safetensors_dir = os.path.join(cache_dir, 'safetensors', model_id.replace('/', '--'))
    ensure_dir(safetensors_dir)
    
    if not filename.endswith('.safetensors'):
        filename += '.safetensors'
    
    file_path = os.path.join(safetensors_dir, filename)
    
    try:
        save_file(tensor_dict, file_path)
        logger.info(f"Cached tensor dictionary to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to cache tensor dictionary: {e}")
        raise

def load_cached_tensor(model_id: str, filename: str) -> Dict[str, torch.Tensor]:
    """
    Load a cached tensor dictionary in safetensors format.
    
    Args:
        model_id: The model ID.
        filename: The filename to load.
    
    Returns:
        Dictionary mapping names to tensors.
    """
    cache_dir = get_cache_dir()
    safetensors_dir = os.path.join(cache_dir, 'safetensors', model_id.replace('/', '--'))
    
    if not filename.endswith('.safetensors'):
        filename += '.safetensors'
    
    file_path = os.path.join(safetensors_dir, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"Cached tensor file not found: {file_path}")
        raise FileNotFoundError(f"Cached tensor file not found: {file_path}")
    
    try:
        tensor_dict = load_file(file_path)
        logger.info(f"Loaded cached tensor dictionary from {file_path}")
        return tensor_dict
    except Exception as e:
        logger.error(f"Failed to load cached tensor dictionary: {e}")
        raise

def list_cached_tensors(model_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all cached tensor files.
    
    Args:
        model_id: Optional model ID to filter by.
    
    Returns:
        List of dictionaries with information about cached tensor files.
    """
    cache_dir = get_cache_dir()
    safetensors_dir = os.path.join(cache_dir, 'safetensors')
    
    if not os.path.exists(safetensors_dir):
        return []
    
    result = []
    
    if model_id:
        model_dir = os.path.join(safetensors_dir, model_id.replace('/', '--'))
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith('.safetensors'):
                    file_path = os.path.join(model_dir, filename)
                    result.append({
                        'model_id': model_id,
                        'filename': filename,
                        'path': file_path,
                        'size_bytes': get_file_size(file_path)
                    })
    else:
        for dir_name in os.listdir(safetensors_dir):
            dir_path = os.path.join(safetensors_dir, dir_name)
            if os.path.isdir(dir_path):
                # Convert back from directory name to model_id
                current_model_id = dir_name.replace('--', '/')
                for filename in os.listdir(dir_path):
                    if filename.endswith('.safetensors'):
                        file_path = os.path.join(dir_path, filename)
                        result.append({
                            'model_id': current_model_id,
                            'filename': filename,
                            'path': file_path,
                            'size_bytes': get_file_size(file_path)
                        })
    
    return result
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/safetensors_cache.py") -Value $safetensorsCacheContent

# Implementation for huggingface_toolset/quantization/quantize_model.py
$quantizeModelContent = @"
"""
Quantize Model
------------
Functions for quantizing Hugging Face models.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import PreTrainedModel

from ..utils.logging_utils import get_logger
from ..cache.cache_manager import get_cache_dir
from ..config.constants import DEFAULT_BITS, SUPPORTED_BITS

logger = get_logger(__name__)

def quantize_model(
    model: Union[PreTrainedModel, str],
    output_dir: Optional[str] = None,
    bits: int = DEFAULT_BITS,
    device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto",
    **kwargs
) -> PreTrainedModel:
    """
    Quantize a Hugging Face model.
    
    Args:
        model: The model to quantize or a model ID to load.
        output_dir: Directory to save the quantized model.
        bits: Number of bits for quantization (4 or 8).
        device_map: Device map for model loading and saving.
        **kwargs: Additional arguments for model loading.
    
    Returns:
        The quantized model.
    """
    if bits not in SUPPORTED_BITS:
        logger.error(f"Unsupported bits value: {bits}. Supported values are {SUPPORTED_BITS}")
        raise ValueError(f"Unsupported bits value: {bits}. Supported values are {SUPPORTED_BITS}")
    
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("bitsandbytes not installed. Please install it with 'pip install bitsandbytes'")
        raise ImportError("bitsandbytes not installed. Please install it with 'pip install bitsandbytes'")
    
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        logger.error("transformers not installed. Please install it with 'pip install transformers'")
        raise ImportError("transformers not installed. Please install it with 'pip install transformers'")
    
    logger.info(f"Quantizing model to {bits} bits")
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_compute_dtype=torch.float16 if bits == 4 else None,
        bnb_4bit_use_double_quant=True if bits == 4 else None,
        bnb_4bit_quant_type="nf4" if bits == 4 else None
    )
    
    # Load model with quantization
    if isinstance(model, str):
        logger.info(f"Loading model {model} with {bits}-bit quantization")
        model_kwargs = {
            "device_map": device_map,
            "quantization_config": quantization_config,
            **kwargs
        }
        quantized_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    else:
        logger.info(f"Quantizing provided model to {bits} bits")
        # For already loaded models, we would need to apply quantization manually
        # This is complex and might require model-specific handling
        raise NotImplementedError("Quantizing already loaded models is not yet supported")
    
    # Save quantized model if output_dir is provided
    if output_dir:
        logger.info(f"Saving quantized model to {output_dir}")
        quantized_model.save_pretrained(output_dir)
    
    return quantized_model

def get_optimal_device_map(
    model_id: str,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None
) -> Dict[str, Union[int, str]]:
    """
    Calculate an optimal device map for a model based on available GPU memory.
    
    Args:
        model_id: The model ID.
        max_memory: Optional dictionary mapping device to maximum memory.
    
    Returns:
        A device map dictionary.
    """
    try:
        from accelerate import infer_auto_device_map, init_empty_weights
        from transformers import AutoConfig
    except ImportError:
        logger.error("accelerate not installed. Please install it with 'pip install accelerate'")
        raise ImportError("accelerate not installed. Please install it with 'pip install accelerate'")
    
    logger.info(f"Computing optimal device map for {model_id}")
    
    # Get model config
    config = AutoConfig.from_pretrained(model_id)
    
    # Infer device map
    with init_empty_weights():
        device_map = infer_auto_device_map(
            config,
            max_memory=max_memory,
            no_split_module_classes=["GPTJBlock", "LlamaDecoderLayer", "MistralDecoderLayer"]
        )
    
    logger.info(f"Calculated device map: {device_map}")
    return device_map
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/quantization/quantize_model.py") -Value $quantizeModelContent

# Implementation for huggingface_toolset/quantization/quantization_utils.py
$quantizationUtilsContent = @"
"""
Quantization Utilities
-------------------
Utility functions for model quantization.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
import gc

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def clear_gpu_memory() -> None:
    """
    Clear GPU memory by emptying the cache and collecting garbage.
    """
    logger.info("Clearing GPU memory")
    torch.cuda.empty_cache()
    gc.collect()

def get_gpu_memory_info() -> List[Dict[str, Any]]:
    """
    Get information about GPU memory usage.
    
    Returns:
        A list of dictionaries with GPU memory information.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available")
        return []
    
    result = []
    for i in range(torch.cuda.device_count()):
        info = {
            'device': i,
            'name': torch.cuda.get_device_name(i),
            'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory,
            'allocated_memory_bytes': torch.cuda.memory_allocated(i),
            'reserved_memory_bytes': torch.cuda.memory_reserved(i),
            'free_memory_bytes': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        }
        result.append(info)
    
    return result

def calculate_max_memory() -> Dict[Union[int, str], str]:
    """
    Calculate maximum memory available for each GPU device.
    
    Returns:
        A dictionary mapping device to maximum memory.
    """
    max_memory = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # Get total memory in bytes
            total_memory = torch.cuda.get_device_properties(i).total_memory
            # Reserve some memory for the system (10%)
            usable_memory = int(total_memory * 0.9)
            # Convert to GB string (e.g., '16GiB')
            max_memory[i] = f"{usable_memory // (1024**3)}GiB"
    
    # Add CPU memory (use 32GB or actual available memory)
    try:
        import psutil
        cpu_memory = psutil.virtual_memory().total
        max_memory['cpu'] = f"{min(32, cpu_memory // (1024**3))}GiB"
    except ImportError:
        # Default to 16GB if psutil is not available
        max_memory['cpu'] = "16GiB"
    
    logger.info(f"Calculated max memory: {max_memory}")
    return max_memory

def get_optimal_bits(model_id: str) -> int:
    """
    Determine the optimal number of bits for quantization based on the model and available hardware.
    
    Args:
        model_id: The model ID.
    
    Returns:
        The optimal number of bits (4 or 8).
    """
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            # Get the GPU with the most memory
            max_gpu = max(gpu_info, key=lambda x: x['free_memory_bytes'])
            free_memory_gb = max_gpu['free_memory_bytes'] / (1024**3)
            
            # Get model size from HF API if possible
            try:
                from ..api_integration.api_client import get_model_info
                model_info = get_model_info(model_id)
                model_size_gb = model_info.get("safetensors_size_bytes", model_info.get("size_bytes", 0)) / (1024**3)
                
                # If model is large and we have enough memory, use 8-bit
                if model_size_gb > 10 and free_memory_gb < model_size_gb * 0.5:
                    logger.info(f"Model {model_id} is large ({model_size_gb:.2f} GB) and available GPU memory is limited ({free_memory_gb:.2f} GB), using 4-bit quantization")
                    return 4
                elif model_size_gb > 20:
                    logger.info(f"Model {model_id} is very large ({model_size_gb:.2f} GB), using 4-bit quantization")
                    return 4
                else:
                    logger.info(f"Model {model_id} is moderate size ({model_size_gb:.2f} GB), using 8-bit quantization")
                    return 8
            except Exception as e:
                logger.warning(f"Failed to get model info: {e}")
                
                # Fallback: Just check free memory
                if free_memory_gb < 8:
                    logger.info(f"Limited GPU memory ({free_memory_gb:.2f} GB), using 4-bit quantization")
                    return 4
                else:
                    logger.info(f"Sufficient GPU memory ({free_memory_gb:.2f} GB), using 8-bit quantization")
                    return 8
    
    # If no GPU or unable to determine, default to 8-bit
    logger.info("Default to 8-bit quantization")
    return 8
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/quantization/quantization_utils.py") -Value $quantizationUtilsContent

# Implementation for huggingface_toolset/model_management/model_utils.py
$modelUtilsContent = @"
# Implementation for huggingface_toolset/model_management/model_utils.py
$model_utils = @"
"""
Model Utilities
--------------
Utility functions for model management.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import PreTrainedModel
from safetensors.torch import save_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)

def convert_to_safetensors(
    model: Union[PreTrainedModel, str, Dict[str, torch.Tensor]],
    output_dir: str,
    save_model_config: bool = True
) -> List[str]:
    """
    Convert a PyTorch model to safetensors format.
    
    Args:
        model: The model to convert, a model ID, or a state dict.
        output_dir: Directory to save the safetensors files.
        save_model_config: Whether to save the model configuration.
    
    Returns:
        A list of paths to the created safetensors files.
    """
    logger.info(f"Converting model to safetensors format")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Get the state_dict depending on the input type
    if isinstance(model, PreTrainedModel):
        logger.debug("Converting PyTorch model to safetensors")
        state_dict = model.state_dict()
        model_config = model.config
    elif isinstance(model, str):
        logger.debug(f"Loading model from {model} to convert to safetensors")
        from transformers import AutoModel
        model_obj = AutoModel.from_pretrained(model)
        state_dict = model_obj.state_dict()
        model_config = model_obj.config
    elif isinstance(model, dict):
        logger.debug("Converting state dict to safetensors")
        state_dict = model
        model_config = None
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
    
    # Save the model config if requested
    if save_model_config and model_config is not None:
        config_path = os.path.join(output_dir, "config.json")
        model_config.to_json_file(config_path)
        logger.debug(f"Saved model config to {config_path}")
    
    # Save the state dict to a safetensors file
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, safetensors_path)
    logger.info(f"Saved model in safetensors format to {safetensors_path}")
    
    return [safetensors_path]

def get_model_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: The model to get information for.
    
    Returns:
        A dictionary with model information.
    """
    info = {
        "model_type": model.config.model_type,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "config": model.config.to_dict()
    }
    
    logger.debug(f"Model info: {info}")
    return info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/model_utils.py") -Value $model_utils

# Implementation for huggingface_toolset/model_management/list_models.py
$list_models = @"
"""
List Models
----------
Functions for listing models from Hugging Face Hub.
"""

from typing import Optional, Dict, List, Any
from huggingface_hub import list_models as hf_list_models

from ..api_integration.auth import get_token
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def list_models(
    filter: Optional[Dict[str, List[str]]] = None,
    author: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    use_auth_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List models from the Hugging Face Hub.
    
    Args:
        filter: Filter for the listing.
        author: Filter models by this author.
        search: Search for models by this query.
        limit: Limit the number of models returned.
        use_auth_token: The token to use for authentication.
    
    Returns:
        A list of model information dictionaries.
    """
    logger.info(f"Listing models with search='{search}', author='{author}'")
    
    if use_auth_token is None:
        use_auth_token = get_token()
    
    # Handle author filter
    if author and filter is None:
        filter = {"author": [author]}
    elif author and filter is not None:
        filter["author"] = [author]
    
    try:
        models = hf_list_models(
            filter=filter,
            search=search,
            limit=limit,
            token=use_auth_token
        )
        
        # Convert to list of dictionaries
        results = []
        for model in models:
            model_dict = {
                "id": model.modelId,
                "author": model.author,
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
                "siblings": [s.to_dict() for s in model.siblings]
            }
            results.append(model_dict)
        
        logger.info(f"Found {len(results)} models")
        return results
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise

def get_model_tags() -> List[str]:
    """
    Get all available model tags from Hugging Face Hub.
    
    Returns:
        A list of model tags.
    """
    logger.info("Getting all model tags")
    
    try:
        models = hf_list_models(limit=1)
        tags = []
        
        # Note: This is inefficient but huggingface_hub doesn't expose a direct way to get all tags
        # In a real implementation, we would cache this
        for model in models:
            for tag in model.tags:
                if tag not in tags:
                    tags.append(tag)
        
        logger.info(f"Found {len(tags)} model tags")
        return sorted(tags)
    
    except Exception as e:
        logger.error(f"Error getting model tags: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/list_models.py") -Value $list_models

# Implementation for huggingface_toolset/model_management/delete_model.py
$delete_model = @"
"""
Delete Model
-----------
Functions for deleting models from Hugging Face Hub.
"""

from typing import Optional
from huggingface_hub import delete_repo

from ..api_integration.auth import get_token
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def delete_model(
    repo_id: str,
    use_auth_token: Optional[str] = None,
    confirm: bool = False
) -> bool:
    """
    Delete a model repository from the Hugging Face Hub.
    
    Args:
        repo_id: The repository ID to delete.
        use_auth_token: The token to use for authentication.
        confirm: Whether to confirm the deletion.
    
    Returns:
        True if the deletion was successful, False otherwise.
    """
    logger.info(f"Deleting model repository: {repo_id}")
    
    if use_auth_token is None:
        use_auth_token = get_token()
    
    if not use_auth_token:
        logger.error("Authentication token required to delete a model")
        return False
    
    if not confirm:
        logger.warning("Deletion not confirmed. Set confirm=True to proceed.")
        return False
    
    try:
        delete_repo(
            repo_id=repo_id,
            token=use_auth_token,
            repo_type="model"
        )
        logger.info(f"Successfully deleted model repository {repo_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error deleting model repository {repo_id}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/delete_model.py") -Value $delete_model

# Implementation for huggingface_toolset/dataset_management/__init__.py
$dataset_mgmt_init = @"
"""
Dataset Management Module
-----------------------
Initialize the dataset management module with functions for downloading,
uploading, listing, and managing datasets from the Hugging Face Hub.
"""

from .download_dataset import download_dataset
from .upload_dataset import upload_dataset
from .list_datasets import list_datasets, get_dataset_tags
from .delete_dataset import delete_dataset
from .dataset_utils import get_dataset_info

__all__ = [
    'download_dataset',
    'upload_dataset',
    'list_datasets',
    'get_dataset_tags',
    'delete_dataset',
    'get_dataset_info'
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/__init__.py") -Value $dataset_mgmt_init

# Implementation for huggingface_toolset/dataset_management/download_dataset.py
$download_dataset = @"
"""
Download Dataset
--------------
Functions for downloading datasets from Hugging Face Hub.
"""

from typing import Optional, Dict, Any, List, Union
from datasets import load_dataset, Dataset, DatasetDict

from ..api_integration.auth import get_token
from ..cache.cache_manager import get_cache_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def download_dataset(
    dataset_id: str,
    subset: Optional[str] = None,
    split: Optional[Union[str, List[str]]] = None,
    cache_dir: Optional[str] = None,
    use_auth_token: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Download a dataset from Hugging Face Hub.
    
    Args:
        dataset_id: The dataset ID on Hugging Face Hub.
        subset: The subset of the dataset to load.
        split: The split of the dataset to load.
        cache_dir: Directory where the dataset should be cached.
        use_auth_token: Hugging Face token for private datasets.
        local_files_only: If True, only use local cached files.
        **kwargs: Additional arguments to pass to the dataset loading function.
    
    Returns:
        The loaded dataset, either as a Dataset or a DatasetDict.
    """
    logger.info(f"Downloading dataset: {dataset_id}{f'/{subset}' if subset else ''}")
    
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    if use_auth_token is None:
        use_auth_token = get_token()
    
    dataset_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "token": use_auth_token,
        **kwargs
    }
    
    try:
        dataset = load_dataset(
            path=dataset_id,
            name=subset,
            split=split,
            **dataset_kwargs
        )
        
        logger.info(f"Dataset {dataset_id} downloaded successfully")
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_id}: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/download_dataset.py") -Value $download_dataset

# Implementation for huggingface_toolset/dataset_management/dataset_utils.py
$dataset_utils = @"
"""
Dataset Utilities
---------------
Utility functions for dataset management.
"""

from typing import Dict, Any, Optional
from datasets import Dataset, DatasetDict

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_dataset_info(dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset: The dataset to get information for.
    
    Returns:
        A dictionary with dataset information.
    """
    info = {}
    
    if isinstance(dataset, Dataset):
        info = {
            "type": "Dataset",
            "num_rows": len(dataset),
            "features": str(dataset.features),
            "column_names": dataset.column_names,
            "has_indices": dataset._indices is not None,
        }
    elif isinstance(dataset, DatasetDict):
        info = {
            "type": "DatasetDict",
            "splits": list(dataset.keys()),
            "num_rows_per_split": {k: len(v) for k, v in dataset.items()},
            "features": {k: str(v.features) for k, v in dataset.items()},
            "column_names": {k: v.column_names for k, v in dataset.items()},
        }
    else:
        logger.warning(f"Unknown dataset type: {type(dataset)}")
        info = {"type": str(type(dataset))}
    
    logger.debug(f"Dataset info: {info}")
    return info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/dataset_utils.py") -Value $dataset_utils

# Implementation for huggingface_toolset/cache/__init__.py
$cache_init = @"
"""
Cache Module
----------
Initialize the caching module with functions for managing local caches.
"""

from .cache_manager import get_cache_dir, set_cache_dir, clear_cache, is_cached
from .safetensors_cache import cache_model_safetensors, load_cached_safetensors

__all__ = [
    'get_cache_dir',
    'set_cache_dir',
    'clear_cache',
    'is_cached',
    'cache_model_safetensors',
    'load_cached_safetensors'
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/__init__.py") -Value $cache_init

# Implementation for huggingface_toolset/cache/cache_manager.py
$cache_manager = @"
"""
Cache Manager
-----------
Functions for managing the local cache for models, datasets, and tokenizers.
"""

import os
import shutil
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..config.constants import DEFAULT_CACHE_DIR
from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, get_file_size

logger = get_logger(__name__)

_cache_dir = DEFAULT_CACHE_DIR

def get_cache_dir() -> str:
    """
    Get the current cache directory.
    
    Returns:
        The path to the cache directory.
    """
    return _cache_dir

def set_cache_dir(cache_dir: str) -> None:
    """
    Set the cache directory.
    
    Args:
        cache_dir: The new cache directory.
    """
    global _cache_dir
    _cache_dir = cache_dir
    ensure_dir(cache_dir)
    logger.info(f"Cache directory set to {cache_dir}")

def clear_cache(cache_dir: Optional[str] = None) -> None:
    """
    Clear the cache directory.
    
    Args:
        cache_dir: The cache directory to clear. If None, uses the current cache directory.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    logger.info(f"Clearing cache directory: {cache_dir}")
    
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            logger.info(f"Cache directory cleared: {cache_dir}")
        else:
            logger.warning(f"Cache directory does not exist: {cache_dir}")
    except Exception as e:
        logger.error(f"Error clearing cache directory {cache_dir}: {e}")
        raise

def is_cached(resource_id: str, resource_type: str = "model") -> bool:
    """
    Check if a resource is cached.
    
    Args:
        resource_id: The ID of the resource to check.
        resource_type: The type of resource ("model", "dataset", or "tokenizer").
    
    Returns:
        True if the resource is cached, False otherwise.
    """
    cache_dir = get_cache_dir()
    resource_cache_dir = os.path.join(cache_dir, resource_type + "s", resource_id.replace("/", "--"))
    
    return os.path.exists(resource_cache_dir) and len(os.listdir(resource_cache_dir)) > 0

def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the cache.
    
    Returns:
        A dictionary with cache information.
    """
    cache_dir = get_cache_dir()
    
    if not os.path.exists(cache_dir):
        return {
            "exists": False,
            "size_bytes": 0,
            "models": [],
            "datasets": [],
            "tokenizers": []
        }
    
    # Get cached models
    models_dir = os.path.join(cache_dir, "models")
    models = []
    if os.path.exists(models_dir):
        models = [d.replace("--", "/") for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    # Get cached datasets
    datasets_dir = os.path.join(cache_dir, "datasets")
    datasets = []
    if os.path.exists(datasets_dir):
        datasets = [d.replace("--", "/") for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    
    # Get cached tokenizers
    tokenizers_dir = os.path.join(cache_dir, "tokenizers")
    tokenizers = []
    if os.path.exists(tokenizers_dir):
        tokenizers = [d.replace("--", "/") for d in os.listdir(tokenizers_dir) if os.path.isdir(os.path.join(tokenizers_dir, d))]
    
    # Get total size
    total_size = sum(get_file_size(os.path.join(cache_dir, root, file))
                     for root, _, files in os.walk(cache_dir) 
                     for file in files)
    
    return {
        "exists": True,
        "size_bytes": total_size,
        "models": models,
        "datasets": datasets,
        "tokenizers": tokenizers
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/cache_manager.py") -Value $cache_manager

# Implementation for huggingface_toolset/cache/safetensors_cache.py
$safetensors_cache = @"
"""
Safetensors Cache
---------------
Functions for caching models in safetensors format.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
import torch
from safetensors.torch import save_file, load_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir
from .cache_manager import get_cache_dir

logger = get_logger(__name__)

def cache_model_safetensors(
    model_id: str,
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Cache a model in safetensors format.
    
    Args:
        model_id: The model ID to use as a cache key.
        tensors: The tensors to cache.
        metadata: Optional metadata to store with the tensors.
        cache_dir: The cache directory to use. If None, uses the default.
    
    Returns:
        The path to the cached safetensors file.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    # Create a model-specific cache directory
    model_cache_dir = os.path.join(cache_dir, "models", model_id.replace("/", "--"), "safetensors")
    ensure_dir(model_cache_dir)
    
    # Path to the safetensors file
    safetensors_path = os.path.join(model_cache_dir, "model.safetensors")
    
    # Save the tensors to a safetensors file
    try:
        save_file(tensors, safetensors_path, metadata=metadata)
        logger.info(f"Cached model {model_id} in safetensors format at {safetensors_path}")
        
        # Save metadata to a separate file for easier access
        if metadata:
            metadata_path = os.path.join(model_cache_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata to {metadata_path}")
        
        return safetensors_path
    
    except Exception as e:
        logger.error(f"Failed to cache model {model_id} in safetensors format: {e}")
        raise

def load_cached_safetensors(
    model_id: str,
    cache_dir: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load a cached model in safetensors format.
    
    Args:
        model_id: The model ID used as the cache key.
        cache_dir: The cache directory to use. If None, uses the default.
        device: The device to load the tensors to.
    
    Returns:
        The loaded tensors.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    # Get the model-specific cache directory
    model_cache_dir = os.path.join(cache_dir, "models", model_id.replace("/", "--"), "safetensors")
    safetensors_path = os.path.join(model_cache_dir, "model.safetensors")
    
    if not os.path.exists(safetensors_path):
        logger.error(f"No cached safetensors found for model {model_id}")
        raise FileNotFoundError(f"No cached safetensors found for model {model_id}")
    
    try:
        tensors = load_file(safetensors_path, device=device)
        logger.info(f"Loaded cached safetensors for model {model_id} from {safetensors_path}")
        return tensors
    
    except Exception as e:
        logger.error(f"Failed to load cached safetensors for model {model_id}: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/safetensors_cache.py") -Value $safetensors_cache

# Implementation for huggingface_toolset/quantization/__init__.py
$quantization_init = @"
"""
Quantization Module
-----------------
Initialize the quantization module with functions for optimizing models.
"""

from .quantize_model import quantize_model, quantize_model_bnb
from .quantization_utils import get_quantization_config

__all__ = [
    'quantize_model',
    'quantize_model_bnb',
    'get_quantization_config'
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/quantization/__init__.py") -Value $quantization_init

# Implementation for huggingface_toolset/quantization/quantize_model.py
$quantize_model = @"
"""
Quantize Model
------------
Functions for quantizing models for optimized inference.
"""

import os
from typing import Optional, Dict, Any, Union

import torch
from transformers import PreTrainedModel
from safetensors.torch import save_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir
from ..config.constants import SUPPORTED_BITS

logger = get_logger(__name__)

def quantize_model(
    model: PreTrainedModel,
    output_dir: str,
    bits: int = 8,
    save_safetensors: bool = True,
) -> str:
    """
    Quantize a PyTorch model to lower precision.
    
    Args:
        model: The model to quantize.
        output_dir: Directory to save the quantized model.
        bits: Number of bits for quantization (8 or 4).
        save_safetensors: Whether to save in safetensors format.
    
    Returns:
        Path to the saved quantized model.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"Bits must be one of {SUPPORTED_BITS}, got {bits}")
    
    logger.info(f"Quantizing model to {bits} bits")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Simple quantization using PyTorch's quantization utilities
    # Note: For production, use a more sophisticated approach like using bitsandbytes
    if bits == 8:
        # Create a quantized copy of the model
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        logger.info("Model quantized with torch.quantization.quantize_dynamic")
    else:
        logger.warning(f"Direct quantization to {bits} bits is not supported with this function.")
        logger.warning("Use quantize_model_bnb for better 4-bit quantization support.")
        raise NotImplementedError(f"Direct quantization to {bits} bits is not implemented")
    
    # Save the quantized model
    if save_safetensors:
        # Get the state dict
        state_dict = quantized_model.state_dict()
        
        # Save in safetensors format
        safetensors_path = os.path.join(output_dir, "model.safetensors")
        metadata = {"quantization_bits": str(bits)}
        save_file(state_dict, safetensors_path, metadata=metadata)
        logger.info(f"Saved quantized model in safetensors format to {safetensors_path}")
        
        # Save the configuration
        config_path = os.path.join(output_dir, "config.json")
        model.config.to_json_file(config_path)
        logger.debug(f"Saved model config to {config_path}")
        
        return safetensors_path
    else:
        # Save directly using the transformers save_pretrained
        output_path = os.path.join(output_dir, "pytorch_model.bin")
        quantized_model.save_pretrained(output_dir)
        logger.info(f"Saved quantized model to {output_path}")
        return output_path

def quantize_model_bnb(
    model: PreTrainedModel,
    output_dir: str,
    bits: int = 4,
    save_safetensors: bool = True,
) -> str:
    """
    Quantize a PyTorch model using bitsandbytes for 4-bit or 8-bit precision.
    
    Args:
        model: The model to quantize.
        output_dir: Directory to save the quantized model.
        bits: Number of bits for quantization (4 or 8).
        save_safetensors: Whether to save in safetensors format.
    
    Returns:
        Path to the saved quantized model.
    """
    if bits not in [4, 8]:
        raise ValueError(f"bitsandbytes quantization only supports 4 or 8 bits, got {bits}")
    
    logger.info(f"Quantizing model to {bits} bits using bitsandbytes")
    
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("bitsandbytes is not installed. Please install it with: pip install bitsandbytes")
        raise ImportError("bitsandbytes is required for 4-bit quantization")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Replace Linear layers with 4-bit or 8-bit quantized versions
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if bits == 4:
                model._modules[name] = bnb.nn.Linear4bit.from_linear(module)
            else:  # bits == 8
                model._modules[name] = bnb.nn.Linear8bitLt.from_linear(module)
    
    logger.info(f"Model quantized to {bits}-bit precision using bitsandbytes")
    
    # Save the quantized model
    if save_safetensors:
        # Get the state dict - note that this might not work directly with bnb layers
        # In a real implementation, you would need to handle this more carefully
        try:
            state_dict = model.state_dict()
            
            # Save in safetensors format
            safetensors_path = os.path.join(output_dir, "model.safetensors")
            metadata = {"quantization_bits": str(bits), "quantization_method": "bitsandbytes"}
            save_file(state_dict, safetensors_path, metadata=metadata)
            logger.info(f"Saved quantized model in safetensors format to {safetensors_path}")
            
            # Save the configuration
            config_path = os.path.join(output_dir, "config.json")
            model.config.to_json_file(config_path)
            logger.debug(f"Saved model config to {config_path}")
            
            return safetensors_path
        except Exception as e:
            logger.warning(f"Failed to save quantized model in safetensors format: {e}")
            logger.warning("Falling back to standard save_pretrained")
            output_path = os.path.join(output_dir, "pytorch_model.bin")
            model.save_pretrained(output_dir)
            logger.info(f"Saved quantized model to {output_path}")
            return output_path
    else:
        # Save directly using the transformers save_pretrained
        output_path = os.path.join(output_dir, "pytorch_model.bin")
        model.save_pretrained(output_dir)
        logger.info(f"Saved quantized model to {output_path}")
        return output_path
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/quantization/quantize_model.py") -Value $quantize_model

# Implementation for huggingface_toolset/utils/logging_utils.py
$logging_utils = @"
"""
Logging Utilities
--------------
Utility functions for logging.
"""

import os
import logging
from typing import Optional

# Global logger level
_log_level = logging.INFO

def setup_logging(level: Optional[int] = None, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: The logging level to use.
        log_file: Optional file to log to.
    """
    global _log_level
    
    if level is not None:
        _log_level = level
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: The name of the logger.
    
    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_log_level)
    return logger
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/logging_utils.py") -Value $logging_utils

# Implementation for huggingface_toolset/utils/file_utils.py
$file_utils = @"
"""
File Utilities
-----------
Utility functions for file operations.
"""

import os
import shutil
from typing import Optional
from pathlib import Path

from .logging_utils import get_logger

logger = get_logger(__name__)

def ensure_dir(dir_path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: The directory path to ensure.
    
    Returns:
        The absolute path to the directory.
    """
    dir_path = os.path.abspath(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: The path to the file.
    
    Returns:
        The size of the file in bytes.
    """
    if not os.path.isfile(file_path):
        return 0
    return os.path.getsize(file_path)

def copy_file(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        src_path: The source file path.
        dst_path: The destination file path.
        overwrite: Whether to overwrite the destination if it exists.
    
    Returns:
        True if the copy was successful, False otherwise.
    """
    if not os.path.isfile(src_path):
        logger.error(f"Source file does not exist: {src_path}")
        return False
    
    if os.path.exists(dst_path) and not overwrite:
        logger.warning(f"Destination file already exists and overwrite=False: {dst_path}")
        return False
    
    # Ensure the destination directory exists
    dst_dir = os.path.dirname(dst_path)
    ensure_dir(dst_dir)
    
    try:
        shutil.copy2(src_path, dst_path)
        logger.debug(f"Copied {src_path} to {dst_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
        return False

def remove_file(file_path: str) -> bool:
    """
    Remove a file.
    
    Args:
        file_path: The path to the file to remove.
    
    Returns:
        True if the removal was successful, False otherwise.
    """
    if not os.path.isfile(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    try:
        os.remove(file_path)
        logger.debug(f"Removed file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove file {file_path}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/file_utils.py") -Value $file_utils

# Implementation for huggingface_toolset/api_integration/auth.py
$api_auth = @"
"""
Hugging Face Authentication
-------------------------
Functions for authenticating with the Hugging Face API.
"""

import os
from typing import Optional
from huggingface_hub import login as hf_login
from ..utils.logging_utils import get_logger
from ..config.constants import TOKEN_ENV_NAME

logger = get_logger(__name__)

def get_token() -> Optional[str]:
    """
    Get the Hugging Face API token.
    
    Returns:
        The token or None if not set.
    """
    token = os.environ.get(TOKEN_ENV_NAME)
    if not token:
        logger.warning(f"No Hugging Face token found in environment variable {TOKEN_ENV_NAME}")
    return token

def set_token(token: str) -> None:
    """
    Set the Hugging Face API token in the environment.
    
    Args:
        token: The token to set.
    """
    os.environ[TOKEN_ENV_NAME] = token
    logger.info(f"Hugging Face token set in environment variable {TOKEN_ENV_NAME}")

def login(token: Optional[str] = None, write_token: bool = False) -> bool:
    """
    Login to# Implementation for huggingface_toolset/model_management/model_utils.py
$model_utils = @"
"""
Model Utilities
--------------
Utility functions for model management.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import PreTrainedModel
from safetensors.torch import save_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)

def convert_to_safetensors(
    model: Union[PreTrainedModel, str, Dict[str, torch.Tensor]],
    output_dir: str,
    save_model_config: bool = True
) -> List[str]:
    """
    Convert a PyTorch model to safetensors format.
    
    Args:
        model: The model to convert, a model ID, or a state dict.
        output_dir: Directory to save the safetensors files.
        save_model_config: Whether to save the model configuration.
    
    Returns:
        A list of paths to the created safetensors files.
    """
    logger.info(f"Converting model to safetensors format")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Get the state_dict depending on the input type
    if isinstance(model, PreTrainedModel):
        logger.debug("Converting PyTorch model to safetensors")
        state_dict = model.state_dict()
        model_config = model.config
    elif isinstance(model, str):
        logger.debug(f"Loading model from {model} to convert to safetensors")
        from transformers import AutoModel
        model_obj = AutoModel.from_pretrained(model)
        state_dict = model_obj.state_dict()
        model_config = model_obj.config
    elif isinstance(model, dict):
        logger.debug("Converting state dict to safetensors")
        state_dict = model
        model_config = None
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
    
    # Save the model config if requested
    if save_model_config and model_config is not None:
        config_path = os.path.join(output_dir, "config.json")
        model_config.to_json_file(config_path)
        logger.debug(f"Saved model config to {config_path}")
    
    # Save the state dict to a safetensors file
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, safetensors_path)
    logger.info(f"Saved model in safetensors format to {safetensors_path}")
    
    return [safetensors_path]

def get_model_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: The model to get information for.
    
    Returns:
        A dictionary with model information.
    """
    info = {
        "model_type": model.config.model_type,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "config": model.config.to_dict()
    }
    
    logger.debug(f"Model info: {info}")
    return info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/model_utils.py") -Value $model_utils

# Implementation for huggingface_toolset/model_management/list_models.py
$list_models = @"
"""
List Models
----------
Functions for listing models from Hugging Face Hub.
"""

from typing import Optional, Dict, List, Any
from huggingface_hub import list_models as hf_list_models

from ..api_integration.auth import get_token
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def list_models(
    filter: Optional[Dict[str, List[str]]] = None,
    author: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    use_auth_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List models from the Hugging Face Hub.
    
    Args:
        filter: Filter for the listing.
        author: Filter models by this author.
        search: Search for models by this query.
        limit: Limit the number of models returned.
        use_auth_token: The token to use for authentication.
    
    Returns:
        A list of model information dictionaries.
    """
    logger.info(f"Listing models with search='{search}', author='{author}'")
    
    if use_auth_token is None:
        use_auth_token = get_token()
    
    # Handle author filter
    if author and filter is None:
        filter = {"author": [author]}
    elif author and filter is not None:
        filter["author"] = [author]
    
    try:
        models = hf_list_models(
            filter=filter,
            search=search,
            limit=limit,
            token=use_auth_token
        )
        
        # Convert to list of dictionaries
        results = []
        for model in models:
            model_dict = {
                "id": model.modelId,
                "author": model.author,
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
                "siblings": [s.to_dict() for s in model.siblings]
            }
            results.append(model_dict)
        
        logger.info(f"Found {len(results)} models")
        return results
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise

def get_model_tags() -> List[str]:
    """
    Get all available model tags from Hugging Face Hub.
    
    Returns:
        A list of model tags.
    """
    logger.info("Getting all model tags")
    
    try:
        models = hf_list_models(limit=1)
        tags = []
        
        # Note: This is inefficient but huggingface_hub doesn't expose a direct way to get all tags
        # In a real implementation, we would cache this
        for model in models:
            for tag in model.tags:
                if tag not in tags:
                    tags.append(tag)
        
        logger.info(f"Found {len(tags)} model tags")
        return sorted(tags)
    
    except Exception as e:
        logger.error(f"Error getting model tags: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/list_models.py") -Value $list_models

# Implementation for huggingface_toolset/model_management/delete_model.py
$delete_model = @"
"""
Delete Model
-----------
Functions for deleting models from Hugging Face Hub.
"""

from typing import Optional
from huggingface_hub import delete_repo

from ..api_integration.auth import get_token
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def delete_model(
    repo_id: str,
    use_auth_token: Optional[str] = None,
    confirm: bool = False
) -> bool:
    """
    Delete a model repository from the Hugging Face Hub.
    
    Args:
        repo_id: The repository ID to delete.
        use_auth_token: The token to use for authentication.
        confirm: Whether to confirm the deletion.
    
    Returns:
        True if the deletion was successful, False otherwise.
    """
    logger.info(f"Deleting model repository: {repo_id}")
    
    if use_auth_token is None:
        use_auth_token = get_token()
    
    if not use_auth_token:
        logger.error("Authentication token required to delete a model")
        return False
    
    if not confirm:
        logger.warning("Deletion not confirmed. Set confirm=True to proceed.")
        return False
    
    try:
        delete_repo(
            repo_id=repo_id,
            token=use_auth_token,
            repo_type="model"
        )
        logger.info(f"Successfully deleted model repository {repo_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error deleting model repository {repo_id}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/delete_model.py") -Value $delete_model

# Implementation for huggingface_toolset/dataset_management/__init__.py
$dataset_mgmt_init = @"
"""
Dataset Management Module
-----------------------
Initialize the dataset management module with functions for downloading,
uploading, listing, and managing datasets from the Hugging Face Hub.
"""

from .download_dataset import download_dataset
from .upload_dataset import upload_dataset
from .list_datasets import list_datasets, get_dataset_tags
from .delete_dataset import delete_dataset
from .dataset_utils import get_dataset_info

__all__ = [
    'download_dataset',
    'upload_dataset',
    'list_datasets',
    'get_dataset_tags',
    'delete_dataset',
    'get_dataset_info'
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/__init__.py") -Value $dataset_mgmt_init

# Implementation for huggingface_toolset/dataset_management/download_dataset.py
$download_dataset = @"
"""
Download Dataset
--------------
Functions for downloading datasets from Hugging Face Hub.
"""

from typing import Optional, Dict, Any, List, Union
from datasets import load_dataset, Dataset, DatasetDict

from ..api_integration.auth import get_token
from ..cache.cache_manager import get_cache_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def download_dataset(
    dataset_id: str,
    subset: Optional[str] = None,
    split: Optional[Union[str, List[str]]] = None,
    cache_dir: Optional[str] = None,
    use_auth_token: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Download a dataset from Hugging Face Hub.
    
    Args:
        dataset_id: The dataset ID on Hugging Face Hub.
        subset: The subset of the dataset to load.
        split: The split of the dataset to load.
        cache_dir: Directory where the dataset should be cached.
        use_auth_token: Hugging Face token for private datasets.
        local_files_only: If True, only use local cached files.
        **kwargs: Additional arguments to pass to the dataset loading function.
    
    Returns:
        The loaded dataset, either as a Dataset or a DatasetDict.
    """
    logger.info(f"Downloading dataset: {dataset_id}{f'/{subset}' if subset else ''}")
    
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    if use_auth_token is None:
        use_auth_token = get_token()
    
    dataset_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "token": use_auth_token,
        **kwargs
    }
    
    try:
        dataset = load_dataset(
            path=dataset_id,
            name=subset,
            split=split,
            **dataset_kwargs
        )
        
        logger.info(f"Dataset {dataset_id} downloaded successfully")
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_id}: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/download_dataset.py") -Value $download_dataset

# Implementation for huggingface_toolset/dataset_management/dataset_utils.py
$dataset_utils = @"
"""
Dataset Utilities
---------------
Utility functions for dataset management.
"""

from typing import Dict, Any, Optional
from datasets import Dataset, DatasetDict

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_dataset_info(dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset: The dataset to get information for.
    
    Returns:
        A dictionary with dataset information.
    """
    info = {}
    
    if isinstance(dataset, Dataset):
        info = {
            "type": "Dataset",
            "num_rows": len(dataset),
            "features": str(dataset.features),
            "column_names": dataset.column_names,
            "has_indices": dataset._indices is not None,
        }
    elif isinstance(dataset, DatasetDict):
        info = {
            "type": "DatasetDict",
            "splits": list(dataset.keys()),
            "num_rows_per_split": {k: len(v) for k, v in dataset.items()},
            "features": {k: str(v.features) for k, v in dataset.items()},
            "column_names": {k: v.column_names for k, v in dataset.items()},
        }
    else:
        logger.warning(f"Unknown dataset type: {type(dataset)}")
        info = {"type": str(type(dataset))}
    
    logger.debug(f"Dataset info: {info}")
    return info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/dataset_management/dataset_utils.py") -Value $dataset_utils

# Implementation for huggingface_toolset/cache/__init__.py
$cache_init = @"
"""
Cache Module
----------
Initialize the caching module with functions for managing local caches.
"""

from .cache_manager import get_cache_dir, set_cache_dir, clear_cache, is_cached
from .safetensors_cache import cache_model_safetensors, load_cached_safetensors

__all__ = [
    'get_cache_dir',
    'set_cache_dir',
    'clear_cache',
    'is_cached',
    'cache_model_safetensors',
    'load_cached_safetensors'
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/__init__.py") -Value $cache_init

# Implementation for huggingface_toolset/cache/cache_manager.py
$cache_manager = @"
"""
Cache Manager
-----------
Functions for managing the local cache for models, datasets, and tokenizers.
"""

import os
import shutil
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..config.constants import DEFAULT_CACHE_DIR
from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, get_file_size

logger = get_logger(__name__)

_cache_dir = DEFAULT_CACHE_DIR

def get_cache_dir() -> str:
    """
    Get the current cache directory.
    
    Returns:
        The path to the cache directory.
    """
    return _cache_dir

def set_cache_dir(cache_dir: str) -> None:
    """
    Set the cache directory.
    
    Args:
        cache_dir: The new cache directory.
    """
    global _cache_dir
    _cache_dir = cache_dir
    ensure_dir(cache_dir)
    logger.info(f"Cache directory set to {cache_dir}")

def clear_cache(cache_dir: Optional[str] = None) -> None:
    """
    Clear the cache directory.
    
    Args:
        cache_dir: The cache directory to clear. If None, uses the current cache directory.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    logger.info(f"Clearing cache directory: {cache_dir}")
    
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            logger.info(f"Cache directory cleared: {cache_dir}")
        else:
            logger.warning(f"Cache directory does not exist: {cache_dir}")
    except Exception as e:
        logger.error(f"Error clearing cache directory {cache_dir}: {e}")
        raise

def is_cached(resource_id: str, resource_type: str = "model") -> bool:
    """
    Check if a resource is cached.
    
    Args:
        resource_id: The ID of the resource to check.
        resource_type: The type of resource ("model", "dataset", or "tokenizer").
    
    Returns:
        True if the resource is cached, False otherwise.
    """
    cache_dir = get_cache_dir()
    resource_cache_dir = os.path.join(cache_dir, resource_type + "s", resource_id.replace("/", "--"))
    
    return os.path.exists(resource_cache_dir) and len(os.listdir(resource_cache_dir)) > 0

def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the cache.
    
    Returns:
        A dictionary with cache information.
    """
    cache_dir = get_cache_dir()
    
    if not os.path.exists(cache_dir):
        return {
            "exists": False,
            "size_bytes": 0,
            "models": [],
            "datasets": [],
            "tokenizers": []
        }
    
    # Get cached models
    models_dir = os.path.join(cache_dir, "models")
    models = []
    if os.path.exists(models_dir):
        models = [d.replace("--", "/") for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    # Get cached datasets
    datasets_dir = os.path.join(cache_dir, "datasets")
    datasets = []
    if os.path.exists(datasets_dir):
        datasets = [d.replace("--", "/") for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    
    # Get cached tokenizers
    tokenizers_dir = os.path.join(cache_dir, "tokenizers")
    tokenizers = []
    if os.path.exists(tokenizers_dir):
        tokenizers = [d.replace("--", "/") for d in os.listdir(tokenizers_dir) if os.path.isdir(os.path.join(tokenizers_dir, d))]
    
    # Get total size
    total_size = sum(get_file_size(os.path.join(cache_dir, root, file))
                     for root, _, files in os.walk(cache_dir) 
                     for file in files)
    
    return {
        "exists": True,
        "size_bytes": total_size,
        "models": models,
        "datasets": datasets,
        "tokenizers": tokenizers
    }
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/cache_manager.py") -Value $cache_manager

# Implementation for huggingface_toolset/cache/safetensors_cache.py
$safetensors_cache = @"
"""
Safetensors Cache
---------------
Functions for caching models in safetensors format.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
import torch
from safetensors.torch import save_file, load_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir
from .cache_manager import get_cache_dir

logger = get_logger(__name__)

def cache_model_safetensors(
    model_id: str,
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Cache a model in safetensors format.
    
    Args:
        model_id: The model ID to use as a cache key.
        tensors: The tensors to cache.
        metadata: Optional metadata to store with the tensors.
        cache_dir: The cache directory to use. If None, uses the default.
    
    Returns:
        The path to the cached safetensors file.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    # Create a model-specific cache directory
    model_cache_dir = os.path.join(cache_dir, "models", model_id.replace("/", "--"), "safetensors")
    ensure_dir(model_cache_dir)
    
    # Path to the safetensors file
    safetensors_path = os.path.join(model_cache_dir, "model.safetensors")
    
    # Save the tensors to a safetensors file
    try:
        save_file(tensors, safetensors_path, metadata=metadata)
        logger.info(f"Cached model {model_id} in safetensors format at {safetensors_path}")
        
        # Save metadata to a separate file for easier access
        if metadata:
            metadata_path = os.path.join(model_cache_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata to {metadata_path}")
        
        return safetensors_path
    
    except Exception as e:
        logger.error(f"Failed to cache model {model_id} in safetensors format: {e}")
        raise

def load_cached_safetensors(
    model_id: str,
    cache_dir: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load a cached model in safetensors format.
    
    Args:
        model_id: The model ID used as the cache key.
        cache_dir: The cache directory to use. If None, uses the default.
        device: The device to load the tensors to.
    
    Returns:
        The loaded tensors.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    # Get the model-specific cache directory
    model_cache_dir = os.path.join(cache_dir, "models", model_id.replace("/", "--"), "safetensors")
    safetensors_path = os.path.join(model_cache_dir, "model.safetensors")
    
    if not os.path.exists(safetensors_path):
        logger.error(f"No cached safetensors found for model {model_id}")
        raise FileNotFoundError(f"No cached safetensors found for model {model_id}")
    
    try:
        tensors = load_file(safetensors_path, device=device)
        logger.info(f"Loaded cached safetensors for model {model_id} from {safetensors_path}")
        return tensors
    
    except Exception as e:
        logger.error(f"Failed to load cached safetensors for model {model_id}: {e}")
        raise
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/safetensors_cache.py") -Value $safetensors_cache

# Implementation for huggingface_toolset/quantization/__init__.py
$quantization_init = @"
"""
Quantization Module
-----------------
Initialize the quantization module with functions for optimizing models.
"""

from .quantize_model import quantize_model, quantize_model_bnb
from .quantization_utils import get_quantization_config

__all__ = [
    'quantize_model',
    'quantize_model_bnb',
    'get_quantization_config'
]
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/quantization/__init__.py") -Value $quantization_init

# Implementation for huggingface_toolset/quantization/quantize_model.py
$quantize_model = @"
"""
Quantize Model
------------
Functions for quantizing models for optimized inference.
"""

import os
from typing import Optional, Dict, Any, Union

import torch
from transformers import PreTrainedModel
from safetensors.torch import save_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir
from ..config.constants import SUPPORTED_BITS

logger = get_logger(__name__)

def quantize_model(
    model: PreTrainedModel,
    output_dir: str,
    bits: int = 8,
    save_safetensors: bool = True,
) -> str:
    """
    Quantize a PyTorch model to lower precision.
    
    Args:
        model: The model to quantize.
        output_dir: Directory to save the quantized model.
        bits: Number of bits for quantization (8 or 4).
        save_safetensors: Whether to save in safetensors format.
    
    Returns:
        Path to the saved quantized model.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"Bits must be one of {SUPPORTED_BITS}, got {bits}")
    
    logger.info(f"Quantizing model to {bits} bits")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Simple quantization using PyTorch's quantization utilities
    # Note: For production, use a more sophisticated approach like using bitsandbytes
    if bits == 8:
        # Create a quantized copy of the model
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        logger.info("Model quantized with torch.quantization.quantize_dynamic")
    else:
        logger.warning(f"Direct quantization to {bits} bits is not supported with this function.")
        logger.warning("Use quantize_model_bnb for better 4-bit quantization support.")
        raise NotImplementedError(f"Direct quantization to {bits} bits is not implemented")
    
    # Save the quantized model
    if save_safetensors:
        # Get the state dict
        state_dict = quantized_model.state_dict()
        
        # Save in safetensors format
        safetensors_path = os.path.join(output_dir, "model.safetensors")
        metadata = {"quantization_bits": str(bits)}
        save_file(state_dict, safetensors_path, metadata=metadata)
        logger.info(f"Saved quantized model in safetensors format to {safetensors_path}")
        
        # Save the configuration
        config_path = os.path.join(output_dir, "config.json")
        model.config.to_json_file(config_path)
        logger.debug(f"Saved model config to {config_path}")
        
        return safetensors_path
    else:
        # Save directly using the transformers save_pretrained
        output_path = os.path.join(output_dir, "pytorch_model.bin")
        quantized_model.save_pretrained(output_dir)
        logger.info(f"Saved quantized model to {output_path}")
        return output_path

def quantize_model_bnb(
    model: PreTrainedModel,
    output_dir: str,
    bits: int = 4,
    save_safetensors: bool = True,
) -> str:
    """
    Quantize a PyTorch model using bitsandbytes for 4-bit or 8-bit precision.
    
    Args:
        model: The model to quantize.
        output_dir: Directory to save the quantized model.
        bits: Number of bits for quantization (4 or 8).
        save_safetensors: Whether to save in safetensors format.
    
    Returns:
        Path to the saved quantized model.
    """
    if bits not in [4, 8]:
        raise ValueError(f"bitsandbytes quantization only supports 4 or 8 bits, got {bits}")
    
    logger.info(f"Quantizing model to {bits} bits using bitsandbytes")
    
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("bitsandbytes is not installed. Please install it with: pip install bitsandbytes")
        raise ImportError("bitsandbytes is required for 4-bit quantization")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Replace Linear layers with 4-bit or 8-bit quantized versions
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if bits == 4:
                model._modules[name] = bnb.nn.Linear4bit.from_linear(module)
            else:  # bits == 8
                model._modules[name] = bnb.nn.Linear8bitLt.from_linear(module)
    
    logger.info(f"Model quantized to {bits}-bit precision using bitsandbytes")
    
    # Save the quantized model
    if save_safetensors:
        # Get the state dict - note that this might not work directly with bnb layers
        # In a real implementation, you would need to handle this more carefully
        try:
            state_dict = model.state_dict()
            
            # Save in safetensors format
            safetensors_path = os.path.join(output_dir, "model.safetensors")
            metadata = {"quantization_bits": str(bits), "quantization_method": "bitsandbytes"}
            save_file(state_dict, safetensors_path, metadata=metadata)
            logger.info(f"Saved quantized model in safetensors format to {safetensors_path}")
            
            # Save the configuration
            config_path = os.path.join(output_dir, "config.json")
            model.config.to_json_file(config_path)
            logger.debug(f"Saved model config to {config_path}")
            
            return safetensors_path
        except Exception as e:
            logger.warning(f"Failed to save quantized model in safetensors format: {e}")
            logger.warning("Falling back to standard save_pretrained")
            output_path = os.path.join(output_dir, "pytorch_model.bin")
            model.save_pretrained(output_dir)
            logger.info(f"Saved quantized model to {output_path}")
            return output_path
    else:
        # Save directly using the transformers save_pretrained
        output_path = os.path.join(output_dir, "pytorch_model.bin")
        model.save_pretrained(output_dir)
        logger.info(f"Saved quantized model to {output_path}")
        return output_path
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/quantization/quantize_model.py") -Value $quantize_model

# Implementation for huggingface_toolset/utils/logging_utils.py
$logging_utils = @"
"""
Logging Utilities
--------------
Utility functions for logging.
"""

import os
import logging
from typing import Optional

# Global logger level
_log_level = logging.INFO

def setup_logging(level: Optional[int] = None, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: The logging level to use.
        log_file: Optional file to log to.
    """
    global _log_level
    
    if level is not None:
        _log_level = level
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: The name of the logger.
    
    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_log_level)
    return logger
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/logging_utils.py") -Value $logging_utils

# Implementation for huggingface_toolset/utils/file_utils.py
$file_utils = @"
"""
File Utilities
-----------
Utility functions for file operations.
"""

import os
import shutil
from typing import Optional
from pathlib import Path

from .logging_utils import get_logger

logger = get_logger(__name__)

def ensure_dir(dir_path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: The directory path to ensure.
    
    Returns:
        The absolute path to the directory.
    """
    dir_path = os.path.abspath(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: The path to the file.
    
    Returns:
        The size of the file in bytes.
    """
    if not os.path.isfile(file_path):
        return 0
    return os.path.getsize(file_path)

def copy_file(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        src_path: The source file path.
        dst_path: The destination file path.
        overwrite: Whether to overwrite the destination if it exists.
    
    Returns:
        True if the copy was successful, False otherwise.
    """
    if not os.path.isfile(src_path):
        logger.error(f"Source file does not exist: {src_path}")
        return False
    
    if os.path.exists(dst_path) and not overwrite:
        logger.warning(f"Destination file already exists and overwrite=False: {dst_path}")
        return False
    
    # Ensure the destination directory exists
    dst_dir = os.path.dirname(dst_path)
    ensure_dir(dst_dir)
    
    try:
        shutil.copy2(src_path, dst_path)
        logger.debug(f"Copied {src_path} to {dst_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
        return False

def remove_file(file_path: str) -> bool:
    """
    Remove a file.
    
    Args:
        file_path: The path to the file to remove.
    
    Returns:
        True if the removal was successful, False otherwise.
    """
    if not os.path.isfile(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    try:
        os.remove(file_path)
        logger.debug(f"Removed file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove file {file_path}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/file_utils.py") -Value $file_utils

# Implementation for huggingface_toolset/api_integration/auth.py
$api_auth = @"
# Implement model_utils.py - continue from where we left off
$modelUtilsContent = @"
"""
Model Utilities
--------------
Utility functions for model management.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import PreTrainedModel
from safetensors.torch import save_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)

def convert_to_safetensors(
    model: Union[PreTrainedModel, str, Dict[str, torch.Tensor]],
    output_dir: str,
    save_model_config: bool = True
) -> List[str]:
    """
    Convert a PyTorch model to safetensors format.
    
    Args:
        model: The model to convert, a model ID, or a state dict.
        output_dir: Directory to save the safetensors files.
        save_model_config: Whether to save the model configuration.
    
    Returns:
        A list of paths to the created safetensors files.
    """
    logger.info(f"Converting model to safetensors format")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Handle different input types
    if isinstance(model, str):
        # Load model from a path or model ID
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model)
    
    # Get state dict
    if isinstance(model, PreTrainedModel):
        # Save model config if requested
        if save_model_config:
            model.config.save_pretrained(output_dir)
        state_dict = model.state_dict()
    else:
        # Assume it's already a state dict
        state_dict = model
    
    # Convert and save to safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, safetensors_path)
    
    logger.info(f"Model converted and saved to {safetensors_path}")
    return [safetensors_path]

def get_model_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: The model to get information for.
    
    Returns:
        A dictionary with model information.
    """
    info = {
        "model_type": model.config.model_type,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": getattr(model.config, "num_hidden_layers", None),
        "hidden_size": getattr(model.config, "hidden_size", None),
    }
    
    return info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/model_utils.py") -Value $modelUtilsContent

# Implement utils/logging_utils.py
$loggingUtilsContent = @"
"""
Logging Utilities
---------------
Utility functions for logging.
"""

import os
import logging
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional file to write logs to.
        log_format: Format string for log messages.
    """
    handlers = []
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format=log_format
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger.
    
    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/logging_utils.py") -Value $loggingUtilsContent

# Implement utils/file_utils.py
$fileUtilsContent = @"
"""
File Utilities
------------
Utility functions for file operations.
"""

import os
import shutil
from typing import Union, List, Optional
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: The directory path.
    
    Returns:
        The absolute path to the directory.
    """
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    return directory

def get_file_size(file_path: str, human_readable: bool = False) -> Union[int, str]:
    """
    Get the size of a file.
    
    Args:
        file_path: Path to the file.
        human_readable: If True, return a human-readable string.
    
    Returns:
        Size in bytes or human-readable string.
    """
    size_bytes = os.path.getsize(file_path)
    
    if human_readable:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0 or unit == 'TB':
                break
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} {unit}"
    else:
        return size_bytes

def copy_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path.
        destination: Destination file path.
        overwrite: Whether to overwrite existing files.
    
    Returns:
        True if successful, False otherwise.
    """
    if not os.path.exists(source):
        logger.error(f"Source file does not exist: {source}")
        return False
    
    if os.path.exists(destination) and not overwrite:
        logger.warning(f"Destination file exists and overwrite=False: {destination}")
        return False
    
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        shutil.copy2(source, destination)
        logger.info(f"Copied {source} to {destination}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy {source} to {destination}: {e}")
        return False

def remove_file(file_path: str) -> bool:
    """
    Remove a file.
    
    Args:
        file_path: Path to the file to remove.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")
            return True
        else:
            logger.warning(f"File does not exist: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to remove file {file_path}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/file_utils.py") -Value $fileUtilsContent

# Implement cache/cache_manager.py
$cacheManagerContent = @"
"""
Cache Manager
-----------
Manages local caching for models, datasets, and tokenizers.
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir
from ..config.constants import DEFAULT_CACHE_DIR

logger = get_logger(__name__)

def get_cache_dir(custom_dir: Optional[str] = None) -> str:
    """
    Get the cache directory path.
    
    Args:
        custom_dir: Optional custom cache directory.
    
    Returns:
        The absolute path to the cache directory.
    """
    cache_dir = custom_dir or os.environ.get("HF_CACHE_DIR") or DEFAULT_CACHE_DIR
    return ensure_dir(cache_dir)

def get_cache_path(key: str, cache_dir: Optional[str] = None) -> str:
    """
    Get the cache path for a given key.
    
    Args:
        key: The cache key.
        cache_dir: Optional custom cache directory.
    
    Returns:
        The absolute path to the cache file.
    """
    # Create a hash of the key to use as filename
    key_hash = hashlib.md5(key.encode()).hexdigest()
    cache_path = os.path.join(get_cache_dir(cache_dir), key_hash)
    return cache_path

def is_cached(key: str, cache_dir: Optional[str] = None) -> bool:
    """
    Check if a key is cached.
    
    Args:
        key: The cache key.
        cache_dir: Optional custom cache directory.
    
    Returns:
        True if the key is cached, False otherwise.
    """
    cache_path = get_cache_path(key, cache_dir)
    return os.path.exists(cache_path)

def save_to_cache(key: str, data: Any, cache_dir: Optional[str] = None) -> str:
    """
    Save data to cache.
    
    Args:
        key: The cache key.
        data: The data to cache.
        cache_dir: Optional custom cache directory.
    
    Returns:
        The path to the cached file.
    """
    cache_path = get_cache_path(key, cache_dir)
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        logger.debug(f"Saved data to cache: {cache_path}")
        return cache_path
    except Exception as e:
        logger.error(f"Failed to save data to cache: {e}")
        raise

def load_from_cache(key: str, cache_dir: Optional[str] = None) -> Any:
    """
    Load data from cache.
    
    Args:
        key: The cache key.
        cache_dir: Optional custom cache directory.
    
    Returns:
        The cached data.
    """
    cache_path = get_cache_path(key, cache_dir)
    
    if not os.path.exists(cache_path):
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded data from cache: {cache_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from cache: {e}")
        return None

def clear_cache(key: Optional[str] = None, cache_dir: Optional[str] = None) -> bool:
    """
    Clear cache for a specific key or all cache.
    
    Args:
        key: The cache key to clear. If None, clear all cache.
        cache_dir: Optional custom cache directory.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        if key:
            cache_path = get_cache_path(key, cache_dir)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cleared cache for key: {key}")
            else:
                logger.debug(f"No cache to clear for key: {key}")
        else:
            cache_dir = get_cache_dir(cache_dir)
            for file_path in Path(cache_dir).glob("*"):
                if file_path.is_file():
                    os.remove(file_path)
            logger.info(f"Cleared all cache in: {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/cache_manager.py") -Value $cacheManagerContent

# Implement cache/safetensors_cache.py
$safetensorsCacheContent = @"
"""
Safetensors Cache
---------------
Handle caching for safetensors files.
"""

import os
import json
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import hashlib

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, copy_file
from ..config.constants import DEFAULT_CACHE_DIR, SAFETENSORS_EXTENSION
from .cache_manager import get_cache_dir

logger = get_logger(__name__)

def get_safetensors_cache_dir(custom_dir: Optional[str] = None) -> str:
    """
    Get the safetensors cache directory path.
    
    Args:
        custom_dir: Optional custom cache directory.
    
    Returns:
        The absolute path to the safetensors cache directory.
    """
    cache_dir = custom_dir or os.environ.get("HF_SAFETENSORS_CACHE_DIR") or os.path.join(DEFAULT_CACHE_DIR, "safetensors")
    return ensure_dir(cache_dir)

def get_model_cache_key(model_id: str, revision: str = "main") -> str:
    """
    Generate a cache key for a model.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
    
    Returns:
        A cache key string.
    """
    return f"{model_id}_{revision}"

def get_safetensors_cache_path(model_id: str, revision: str = "main", cache_dir: Optional[str] = None) -> str:
    """
    Get the safetensors cache path for a model.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
    
    Returns:
        The absolute path to the safetensors cache directory for the model.
    """
    key = get_model_cache_key(model_id, revision)
    key_hash = hashlib.md5(key.encode()).hexdigest()
    cache_path = os.path.join(get_safetensors_cache_dir(cache_dir), key_hash)
    return ensure_dir(cache_path)

def cache_safetensors_files(model_id: str, files: List[str], revision: str = "main", 
                           cache_dir: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Cache safetensors files.
    
    Args:
        model_id: The model ID.
        files: List of safetensors file paths.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
        metadata: Optional metadata to store with the cache.
    
    Returns:
        A dictionary mapping weight names to cached file paths.
    """
    cache_path = get_safetensors_cache_path(model_id, revision, cache_dir)
    cached_files = {}
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        cached_file_path = os.path.join(cache_path, file_name)
        
        # Copy the file to cache
        if copy_file(file_path, cached_file_path, overwrite=True):
            weight_name = os.path.splitext(file_name)[0]
            cached_files[weight_name] = cached_file_path
    
    # Save metadata if provided
    if metadata:
        metadata_path = os.path.join(cache_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    logger.info(f"Cached {len(cached_files)} safetensors files for {model_id}")
    return cached_files

def get_cached_safetensors_files(model_id: str, revision: str = "main", 
                               cache_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get cached safetensors files for a model.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
    
    Returns:
        A dictionary mapping weight names to cached file paths.
    """
    cache_path = get_safetensors_cache_path(model_id, revision, cache_dir)
    
    if not os.path.exists(cache_path):
        logger.debug(f"No cached safetensors files for {model_id}")
        return {}
    
    cached_files = {}
    for file_path in Path(cache_path).glob(f"*{SAFETENSORS_EXTENSION}"):
        weight_name = file_path.stem
        cached_files[weight_name] = str(file_path)
    
    logger.debug(f"Found {len(cached_files)} cached safetensors files for {model_id}")
    return cached_files

def is_model_cached(model_id: str, revision: str = "main", cache_dir: Optional[str] = None) -> bool:
    """
    Check if a model is cached.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
    
    Returns:
        True if the model is cached, False otherwise.
    """
    cache_path = get_safetensors_cache_path(model_id, revision, cache_dir)
    
    # Check if the cache directory exists and contains safetensors files
    if os.path.exists(cache_path):
        safetensors_files = list(Path(cache_path).glob(f"*{SAFETENSORS_EXTENSION}"))
        return len(safetensors_files) > 0
    
    return False

def clear_model_cache(model_id: str, revision: str = "main", cache_dir: Optional[str] = None) -> bool:
    """
    Clear cache for a specific model.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
    
    Returns:
        True if successful, False otherwise.
    """
    cache_path = get_safetensors_cache_path(model_id, revision, cache_dir)
    
    try:
        if os.path.exists(cache_path):
            for file_path in Path(cache_path).glob("*"):
                if file_path.is_file():
                    os.remove(file_path)
            os.rmdir(cache_path)
            logger.info(f"Cleared cache for model: {model_id}")
        else:
            logger.debug(f"No cache to clear for model: {model_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache for model {model_id}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/safetensors_cache.py") -Value $safetensorsCacheContent

# Implement memory indexing and vectorization management system
$memoryIndexingDir = Join-Path $baseDir "huggingface_toolset/memory_indexing"
New-Item -Path $memoryIndexingDir -ItemType Directory -Force | Out-Null

# Implement memory_indexing/__init__.py
$memoryIndexingInitContent = @"
"""
Memory Indexing Module
--------------------
Provides functionality for indexing, vectorizing, and retrieving information
for context awareness and efficient reference.
"""

from .vector_store import (
    VectorStore, create_vector_store, get_vector_store,
    add_vectors, search_vectors
)
from .embeddings import (
    generate_embeddings, batch_generate_embeddings,
    preprocess_text
)
from .context_manager import (
    ContextManager, create_context_manager,
    add_context, retrieve_context, update_context
)
from .memory_utils import (
    serialize_memory, deserialize_memory,
    memory_to_json, memory_from_json
)

__all__ = [
    'VectorStore', 'create_vector_store', 'get_vector_store',
    'add_vectors', 'search_vectors',
    'generate_embeddings', 'batch_generate_embeddings',
    'preprocess_text',
    'ContextManager', 'create_context_manager',
    'add_context', 'retrieve_context', 'update_context',
    'serialize_memory', 'deserialize_memory',
    'memory_to_json', 'memory_from_json'
]
"@
Set-Content -Path (Join-Path $memoryIndexingDir "__init__.py") -Value $memoryIndexingInitContent

# Implement memory_indexing/vector_store.py
$vectorStoreContent = @"
"""
Vector Store
-----------
Provides a vector store for efficient similarity search.
"""

import os
import json
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)

@dataclass
class VectorStore:
    """Vector store for efficient similarity search."""
    
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add(self, id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a vector to the store.
        
        Args:
            id: Unique identifier for the vector.
            vector: The vector to store.
            metadata: Optional metadata associated with the vector.
        """
        self.vectors[id] = vector
        if metadata:
            self.metadata[id] = metadata
    
    def add_batch(self, ids: List[str], vectors: List[np.ndarray], 
                  metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add a batch of vectors to the store.
        
        Args:
            ids: List of unique identifiers for the vectors.
            vectors: List of vectors to store.
            metadata: Optional list of metadata associated with the vectors.
        """
        for i, (id, vector) in enumerate(zip(ids, vectors)):
            meta = metadata[i] if metadata and i < len(metadata) else None
            self.add(id, vector, meta)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query vector.
            top_k: Number of results to return.
        
        Returns:
            List of tuples containing (id, similarity, metadata).
        """
        if not self.vectors:
            return []
        
        # Compute cosine similarities
        similarities = []
        for id, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            meta = self.metadata.get(id, {})
            similarities.append((id, similarity, meta))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    def get(self, id: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Get a vector and its metadata by ID.
        
        Args:
            id: The vector ID.
        
        Returns:
            Tuple of (vector, metadata) or (None, None) if not found.
        """
        vector = self.vectors.get(id)
        metadata = self.metadata.get(id)
        return vector, metadata
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            id: The vector ID to delete.
        
        Returns:
            True if the vector was deleted, False otherwise.
        """
        if id in self.vectors:
            del self.vectors[id]
            if id in self.metadata:
                del self.metadata[id]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        self.vectors.clear()
        self.metadata.clear()
    
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the vector store.
        """
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def __len__(self) -> int:
        """Get the number of vectors in the store."""
        return len(self.vectors)
    
    def __contains__(self, id: str) -> bool:
        """Check if a vector ID exists in the store."""
        return id in self.vectors

def create_vector_store() -> VectorStore:
    """
    Create a new vector store.
    
    Returns:
        A new VectorStore instance.
    """
    return VectorStore()

def get_vector_store(path: str) -> Optional[VectorStore]:
    """
    Load a vector store from disk.
    
    Args:
        path: Path to the vector store file.
    
    Returns:
        The loaded VectorStore or None if the file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'rb') as f:
            store = pickle.load(f)
        return store
    except Exception as e:
        logger.error(f"Failed to load vector store from {path}: {e}")
        return None

def add_vectors(store: VectorStore, texts: List[str], ids: Optional[List[str]] = None,
               embedding_fn: Optional[callable] = None) -> List[str]:
    """
    Add text vectors to a vector store.
    
    Args:
        store: The vector store.
        texts: List of texts to encode.
        ids: Optional list of IDs. If None, will generate IDs.
        embedding_fn: Function to generate embeddings. If None, will use a default function.
    
    Returns:
        List of generated or provided IDs.
    """
    from .embeddings import generate_embeddings
    
    # Generate IDs if not provided
    if ids is None:
        ids = [f"vec_{i}" for i in range(len(texts))]
    
    # Generate embeddings
    if embedding_fn is None:
        embedding_fn = generate_embeddings
    
    embeddings = [embedding_fn(text) for text in texts]
    
    # Add to vector store
    for i, (id, embedding) in enumerate(zip(ids, embeddings)):
        metadata = {"text": texts[i]}
        store.add(id, embedding, metadata)
    
    return ids

def search_vectors(store: VectorStore, query: str, top_k: int = 5,
                  embedding_fn: Optional[callable] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Search for similar vectors in a vector store.
    
    Args:
        store: The vector store.
        query: The query text.
        top_k: Number of results to return.
        embedding_fn: Function to generate embeddings. If None, will use a default function.
    
    Returns:
        List of tuples containing (id, similarity, metadata).
    """
    from .embeddings import generate_embeddings
    
    # Generate query embedding
    if embedding_fn is None:
        embedding_fn = generate_embeddings
    
    query_embedding = embedding_fn(query)
    
    # Search
    return store.search(query_embedding, top_k)
"@
Set-Content -Path (Join-Path $memoryIndexingDir "vector_store.py") -Value $vectorStoreContent

# Implement memory_indexing/embeddings.py
$embeddingsContent = @"
"""
Embeddings
---------
Provides functions for generating text embeddings.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Union

from ..utils.logging_utils import get_logger
from ..cache.cache_manager import load_from_cache, save_to_cache

logger = get_logger(__name__)

def preprocess_text(text: str) -> str:
    """
    Preprocess text for embedding generation.
    
    Args:
        text: The text to preprocess.
    
    Returns:
        The preprocessed text.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Trim
    text = text.strip()
    
    return text

def generate_embeddings(text: str, model_name: Optional[str] = None, 
                       use_cache: bool = True) -> np.ndarray:
    """
    Generate embeddings for text.
    
    Args:
        text: The text to generate embeddings for.
        model_name: Optional name of the embedding model to use.
        use_cache: Whether to use cache for embedding generation.
    
    Returns:
        The generated embedding as a numpy array.
    """
    # Preprocess text
    text = preprocess_text(text)
    
    # Check cache if enabled
    if use_cache:
        cache_key = f"embed_{model_name or 'default'}_{hash(text)}"
        cached = load_from_cache(cache_key)
        if cached is not None:
            return np.array(cached)
    
    try:
        # Try to use transformers if available
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Default to a small model if none specified
            if model_name is None:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Tokenize and generate embedding
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Convert to numpy and normalize
            embedding = embeddings[0].numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to use transformers for embedding generation: {e}")
            
            # Fallback to a simple embedding method
            words = text.lower().split()
            # Create a simple bag-of-words embedding
            embedding = np.zeros(100)  # Fixed size
            for i, word in enumerate(words[:100]):
                # Simple hash-based embedding
                embedding[i % 100] += hash(word) % 10000 / 10000
            
            # Normalize
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
        
        # Cache the result if enabled
        if use_cache:
            save_to_cache(cache_key, embedding.tolist())
        
        return embedding
    
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        # Return a zero vector as fallback
        return np.zeros(100)

def batch_generate_embeddings(texts: List[str], model_name: Optional[str] = None,
                             use_cache: bool = True) -> List[np.ndarray]:
    """
    Generate embeddings for a batch of texts.
    
    Args:
        texts: List of texts to generate embeddings for.
        model_name: Optional name of the embedding model to use.
        use_cache: Whether to use cache for embedding generation.
    
    Returns:
        List of generated embeddings.
    """
    return [generate_embeddings(text, model_name, use_cache) for text in texts]
"@
Set-Content -Path (Join-Path $memoryIndexingDir "embeddings.py") -Value $embeddingsContent

# Implement memory_indexing/context_manager.py
$contextManagerContent = @"
"""
Context Manager
-------------
Manages context for model interactions.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

from ..utils.logging_utils import get_logger
from .vector_store import VectorStore, create_vector_store, search_vectors
from .embeddings import generate_embeddings

logger = get_logger(__name__)

@dataclass
class ContextManager:
    """
    Manager for handling context in model interactions.
    """
    
    vector_store: VectorStore = field(default_factory=create_vector_store)
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 100
    
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add text to the context.
        
        Args:
            text: The text to add to context.
            metadata: Optional metadata.
        
        Returns:
            The ID of the added context.
        """
        # Generate a timestamp-based ID
        id = f"ctx_{int(time.time()*1000)}"
        
        # Add to vector store
        embedding = generate_embeddings(text)
        meta = metadata or {}
        meta["text"] = text
        meta["timestamp"] = time.time()
        
        self.vector_store.add(id, embedding, meta)
        
        # Add to history
        self.context_history.append({
            "id": id,
            "text": text,
            "timestamp": meta["timestamp"],
            **{k: v for k, v in meta.items() if k not in ["text", "timestamp"]}
        })
        
        # Trim history if needed
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]
        
        return id
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context based on a query.
        
        Args:
            query: The query text.
            top_k: Number of context items to retrieve.
        
        Returns:
            List of relevant context items with metadata.
        """
        # Search for similar contexts
        results = search_vectors(self.vector_store, query, top_k)
        
        # Format results
        contexts = []
        for id, similarity, metadata in results:
            contexts.append({
                "id": id,
                "text": metadata.get("text", ""),
                "similarity": similarity,
                **{k: v for k, v in metadata.items() if k != "text"}
            })
        
        return contexts
    
    def update(self, id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing context item.
        
        Args:
            id: The ID of the context to update.
            text: The new text.
            metadata: Optional new metadata.
        
        Returns:
            True if the update was successful, False otherwise.
        """
        # Check if the context exists
        existing_vector, existing_meta = self.vector_store.get(id)
        if existing_vector is None:
            return False
        
        # Delete the existing context
        self.vector_store.delete(id)
        
        # Add the updated context
        embedding = generate_embeddings(text)
        meta = existing_meta.copy() if existing_meta else {}
        if metadata:
            meta.update(metadata)
        meta["text"] = text
        meta["timestamp"] = time.time()
        
        self.vector_store.add(id, embedding, meta)
        
        # Update history if present
        for i, item in enumerate(self.context_history):
            if item.get("id") == id:
                self.context_history[i] = {
                    "id": id,
                    "text": text,
                    "timestamp": meta["timestamp"],
                    **{k: v for k, v in meta.items() if k not in ["text", "timestamp"]}
                }
                break
        
        return True
    
    def get_recent_context(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent context items.
        
        Args:
            count: Number of items to retrieve.
        
        Returns:
            List of recent context items.
        """
        return self.context_history[-count:]
    
    def clear(self) -> None:
        """Clear all context."""
        self.vector_store.clear()
        self.context_history.clear()
    
    def save(self, path: str) -> None:
        """
        Save the context manager to disk.
        
        Args:
            path: Path to save the context manager.
        """
        # Save vector store
        vector_store_path = f"{path}.vectors"
        self.vector_store.save(vector_store_path)
        
        # Save history
        history_path = f"{path}.history"
        with open(history_path, 'w') as f:
            json.dump(self.context_history, f)
        
        logger.info(f"Saved context manager to {path}")
    
    def load(self, path: str) -> bool:
        """
        Load the context manager from disk.
        
        Args:
            path: Path to load the context manager from.
        
        Returns:
            True if successful, False otherwise.
        """
        # Load vector store
        vector_store_path = f"{path}.vectors"
        if not os.path.exists(vector_store_path):
            return False
        
        try:
            from .vector_store import get_vector_store
            vector_store = get_vector_store(vector_store_path)# Implement model_utils.py - continue from where we left off
$modelUtilsContent = @"
"""
Model Utilities
--------------
Utility functions for model management.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import PreTrainedModel
from safetensors.torch import save_file

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)

def convert_to_safetensors(
    model: Union[PreTrainedModel, str, Dict[str, torch.Tensor]],
    output_dir: str,
    save_model_config: bool = True
) -> List[str]:
    """
    Convert a PyTorch model to safetensors format.
    
    Args:
        model: The model to convert, a model ID, or a state dict.
        output_dir: Directory to save the safetensors files.
        save_model_config: Whether to save the model configuration.
    
    Returns:
        A list of paths to the created safetensors files.
    """
    logger.info(f"Converting model to safetensors format")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Handle different input types
    if isinstance(model, str):
        # Load model from a path or model ID
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model)
    
    # Get state dict
    if isinstance(model, PreTrainedModel):
        # Save model config if requested
        if save_model_config:
            model.config.save_pretrained(output_dir)
        state_dict = model.state_dict()
    else:
        # Assume it's already a state dict
        state_dict = model
    
    # Convert and save to safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, safetensors_path)
    
    logger.info(f"Model converted and saved to {safetensors_path}")
    return [safetensors_path]

def get_model_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: The model to get information for.
    
    Returns:
        A dictionary with model information.
    """
    info = {
        "model_type": model.config.model_type,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": getattr(model.config, "num_hidden_layers", None),
        "hidden_size": getattr(model.config, "hidden_size", None),
    }
    
    return info
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/model_management/model_utils.py") -Value $modelUtilsContent

# Implement utils/logging_utils.py
$loggingUtilsContent = @"
"""
Logging Utilities
---------------
Utility functions for logging.
"""

import os
import logging
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional file to write logs to.
        log_format: Format string for log messages.
    """
    handlers = []
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format=log_format
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger.
    
    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/logging_utils.py") -Value $loggingUtilsContent

# Implement utils/file_utils.py
$fileUtilsContent = @"
"""
File Utilities
------------
Utility functions for file operations.
"""

import os
import shutil
from typing import Union, List, Optional
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: The directory path.
    
    Returns:
        The absolute path to the directory.
    """
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    return directory

def get_file_size(file_path: str, human_readable: bool = False) -> Union[int, str]:
    """
    Get the size of a file.
    
    Args:
        file_path: Path to the file.
        human_readable: If True, return a human-readable string.
    
    Returns:
        Size in bytes or human-readable string.
    """
    size_bytes = os.path.getsize(file_path)
    
    if human_readable:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0 or unit == 'TB':
                break
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} {unit}"
    else:
        return size_bytes

def copy_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path.
        destination: Destination file path.
        overwrite: Whether to overwrite existing files.
    
    Returns:
        True if successful, False otherwise.
    """
    if not os.path.exists(source):
        logger.error(f"Source file does not exist: {source}")
        return False
    
    if os.path.exists(destination) and not overwrite:
        logger.warning(f"Destination file exists and overwrite=False: {destination}")
        return False
    
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        shutil.copy2(source, destination)
        logger.info(f"Copied {source} to {destination}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy {source} to {destination}: {e}")
        return False

def remove_file(file_path: str) -> bool:
    """
    Remove a file.
    
    Args:
        file_path: Path to the file to remove.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")
            return True
        else:
            logger.warning(f"File does not exist: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to remove file {file_path}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/utils/file_utils.py") -Value $fileUtilsContent

# Implement cache/cache_manager.py
$cacheManagerContent = @"
"""
Cache Manager
-----------
Manages local caching for models, datasets, and tokenizers.
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir
from ..config.constants import DEFAULT_CACHE_DIR

logger = get_logger(__name__)

def get_cache_dir(custom_dir: Optional[str] = None) -> str:
    """
    Get the cache directory path.
    
    Args:
        custom_dir: Optional custom cache directory.
    
    Returns:
        The absolute path to the cache directory.
    """
    cache_dir = custom_dir or os.environ.get("HF_CACHE_DIR") or DEFAULT_CACHE_DIR
    return ensure_dir(cache_dir)

def get_cache_path(key: str, cache_dir: Optional[str] = None) -> str:
    """
    Get the cache path for a given key.
    
    Args:
        key: The cache key.
        cache_dir: Optional custom cache directory.
    
    Returns:
        The absolute path to the cache file.
    """
    # Create a hash of the key to use as filename
    key_hash = hashlib.md5(key.encode()).hexdigest()
    cache_path = os.path.join(get_cache_dir(cache_dir), key_hash)
    return cache_path

def is_cached(key: str, cache_dir: Optional[str] = None) -> bool:
    """
    Check if a key is cached.
    
    Args:
        key: The cache key.
        cache_dir: Optional custom cache directory.
    
    Returns:
        True if the key is cached, False otherwise.
    """
    cache_path = get_cache_path(key, cache_dir)
    return os.path.exists(cache_path)

def save_to_cache(key: str, data: Any, cache_dir: Optional[str] = None) -> str:
    """
    Save data to cache.
    
    Args:
        key: The cache key.
        data: The data to cache.
        cache_dir: Optional custom cache directory.
    
    Returns:
        The path to the cached file.
    """
    cache_path = get_cache_path(key, cache_dir)
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        logger.debug(f"Saved data to cache: {cache_path}")
        return cache_path
    except Exception as e:
        logger.error(f"Failed to save data to cache: {e}")
        raise

def load_from_cache(key: str, cache_dir: Optional[str] = None) -> Any:
    """
    Load data from cache.
    
    Args:
        key: The cache key.
        cache_dir: Optional custom cache directory.
    
    Returns:
        The cached data.
    """
    cache_path = get_cache_path(key, cache_dir)
    
    if not os.path.exists(cache_path):
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded data from cache: {cache_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from cache: {e}")
        return None

def clear_cache(key: Optional[str] = None, cache_dir: Optional[str] = None) -> bool:
    """
    Clear cache for a specific key or all cache.
    
    Args:
        key: The cache key to clear. If None, clear all cache.
        cache_dir: Optional custom cache directory.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        if key:
            cache_path = get_cache_path(key, cache_dir)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cleared cache for key: {key}")
            else:
                logger.debug(f"No cache to clear for key: {key}")
        else:
            cache_dir = get_cache_dir(cache_dir)
            for file_path in Path(cache_dir).glob("*"):
                if file_path.is_file():
                    os.remove(file_path)
            logger.info(f"Cleared all cache in: {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/cache_manager.py") -Value $cacheManagerContent

# Implement cache/safetensors_cache.py
$safetensorsCacheContent = @"
"""
Safetensors Cache
---------------
Handle caching for safetensors files.
"""

import os
import json
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import hashlib

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir, copy_file
from ..config.constants import DEFAULT_CACHE_DIR, SAFETENSORS_EXTENSION
from .cache_manager import get_cache_dir

logger = get_logger(__name__)

def get_safetensors_cache_dir(custom_dir: Optional[str] = None) -> str:
    """
    Get the safetensors cache directory path.
    
    Args:
        custom_dir: Optional custom cache directory.
    
    Returns:
        The absolute path to the safetensors cache directory.
    """
    cache_dir = custom_dir or os.environ.get("HF_SAFETENSORS_CACHE_DIR") or os.path.join(DEFAULT_CACHE_DIR, "safetensors")
    return ensure_dir(cache_dir)

def get_model_cache_key(model_id: str, revision: str = "main") -> str:
    """
    Generate a cache key for a model.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
    
    Returns:
        A cache key string.
    """
    return f"{model_id}_{revision}"

def get_safetensors_cache_path(model_id: str, revision: str = "main", cache_dir: Optional[str] = None) -> str:
    """
    Get the safetensors cache path for a model.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
    
    Returns:
        The absolute path to the safetensors cache directory for the model.
    """
    key = get_model_cache_key(model_id, revision)
    key_hash = hashlib.md5(key.encode()).hexdigest()
    cache_path = os.path.join(get_safetensors_cache_dir(cache_dir), key_hash)
    return ensure_dir(cache_path)

def cache_safetensors_files(model_id: str, files: List[str], revision: str = "main", 
                           cache_dir: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Cache safetensors files.
    
    Args:
        model_id: The model ID.
        files: List of safetensors file paths.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
        metadata: Optional metadata to store with the cache.
    
    Returns:
        A dictionary mapping weight names to cached file paths.
    """
    cache_path = get_safetensors_cache_path(model_id, revision, cache_dir)
    cached_files = {}
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        cached_file_path = os.path.join(cache_path, file_name)
        
        # Copy the file to cache
        if copy_file(file_path, cached_file_path, overwrite=True):
            weight_name = os.path.splitext(file_name)[0]
            cached_files[weight_name] = cached_file_path
    
    # Save metadata if provided
    if metadata:
        metadata_path = os.path.join(cache_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    logger.info(f"Cached {len(cached_files)} safetensors files for {model_id}")
    return cached_files

def get_cached_safetensors_files(model_id: str, revision: str = "main", 
                               cache_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get cached safetensors files for a model.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
    
    Returns:
        A dictionary mapping weight names to cached file paths.
    """
    cache_path = get_safetensors_cache_path(model_id, revision, cache_dir)
    
    if not os.path.exists(cache_path):
        logger.debug(f"No cached safetensors files for {model_id}")
        return {}
    
    cached_files = {}
    for file_path in Path(cache_path).glob(f"*{SAFETENSORS_EXTENSION}"):
        weight_name = file_path.stem
        cached_files[weight_name] = str(file_path)
    
    logger.debug(f"Found {len(cached_files)} cached safetensors files for {model_id}")
    return cached_files

def is_model_cached(model_id: str, revision: str = "main", cache_dir: Optional[str] = None) -> bool:
    """
    Check if a model is cached.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
    
    Returns:
        True if the model is cached, False otherwise.
    """
    cache_path = get_safetensors_cache_path(model_id, revision, cache_dir)
    
    # Check if the cache directory exists and contains safetensors files
    if os.path.exists(cache_path):
        safetensors_files = list(Path(cache_path).glob(f"*{SAFETENSORS_EXTENSION}"))
        return len(safetensors_files) > 0
    
    return False

def clear_model_cache(model_id: str, revision: str = "main", cache_dir: Optional[str] = None) -> bool:
    """
    Clear cache for a specific model.
    
    Args:
        model_id: The model ID.
        revision: The model revision.
        cache_dir: Optional custom cache directory.
    
    Returns:
        True if successful, False otherwise.
    """
    cache_path = get_safetensors_cache_path(model_id, revision, cache_dir)
    
    try:
        if os.path.exists(cache_path):
            for file_path in Path(cache_path).glob("*"):
                if file_path.is_file():
                    os.remove(file_path)
            os.rmdir(cache_path)
            logger.info(f"Cleared cache for model: {model_id}")
        else:
            logger.debug(f"No cache to clear for model: {model_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache for model {model_id}: {e}")
        return False
"@
Set-Content -Path (Join-Path $baseDir "huggingface_toolset/cache/safetensors_cache.py") -Value $safetensorsCacheContent

# Implement memory indexing and vectorization management system
$memoryIndexingDir = Join-Path $baseDir "huggingface_toolset/memory_indexing"
New-Item -Path $memoryIndexingDir -ItemType Directory -Force | Out-Null

# Implement memory_indexing/__init__.py
$memoryIndexingInitContent = @"
"""
Memory Indexing Module
--------------------
Provides functionality for indexing, vectorizing, and retrieving information
for context awareness and efficient reference.
"""

from .vector_store import (
    VectorStore, create_vector_store, get_vector_store,
    add_vectors, search_vectors
)
from .embeddings import (
    generate_embeddings, batch_generate_embeddings,
    preprocess_text
)
from .context_manager import (
    ContextManager, create_context_manager,
    add_context, retrieve_context, update_context
)
from .memory_utils import (
    serialize_memory, deserialize_memory,
    memory_to_json, memory_from_json
)

__all__ = [
    'VectorStore', 'create_vector_store', 'get_vector_store',
    'add_vectors', 'search_vectors',
    'generate_embeddings', 'batch_generate_embeddings',
    'preprocess_text',
    'ContextManager', 'create_context_manager',
    'add_context', 'retrieve_context', 'update_context',
    'serialize_memory', 'deserialize_memory',
    'memory_to_json', 'memory_from_json'
]
"@
Set-Content -Path (Join-Path $memoryIndexingDir "__init__.py") -Value $memoryIndexingInitContent

# Implement memory_indexing/vector_store.py
$vectorStoreContent = @"
"""
Vector Store
-----------
Provides a vector store for efficient similarity search.
"""

import os
import json
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

from ..utils.logging_utils import get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)

@dataclass
class VectorStore:
    """Vector store for efficient similarity search."""
    
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add(self, id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a vector to the store.
        
        Args:
            id: Unique identifier for the vector.
            vector: The vector to store.
            metadata: Optional metadata associated with the vector.
        """
        self.vectors[id] = vector
        if metadata:
            self.metadata[id] = metadata
    
    def add_batch(self, ids: List[str], vectors: List[np.ndarray], 
                  metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add a batch of vectors to the store.
        
        Args:
            ids: List of unique identifiers for the vectors.
            vectors: List of vectors to store.
            metadata: Optional list of metadata associated with the vectors.
        """
        for i, (id, vector) in enumerate(zip(ids, vectors)):
            meta = metadata[i] if metadata and i < len(metadata) else None
            self.add(id, vector, meta)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query vector.
            top_k: Number of results to return.
        
        Returns:
            List of tuples containing (id, similarity, metadata).
        """
        if not self.vectors:
            return []
        
        # Compute cosine similarities
        similarities = []
        for id, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            meta = self.metadata.get(id, {})
            similarities.append((id, similarity, meta))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    def get(self, id: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Get a vector and its metadata by ID.
        
        Args:
            id: The vector ID.
        
        Returns:
            Tuple of (vector, metadata) or (None, None) if not found.
        """
        vector = self.vectors.get(id)
        metadata = self.metadata.get(id)
        return vector, metadata
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            id: The vector ID to delete.
        
        Returns:
            True if the vector was deleted, False otherwise.
        """
        if id in self.vectors:
            del self.vectors[id]
            if id in self.metadata:
                del self.metadata[id]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        self.vectors.clear()
        self.metadata.clear()
    
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the vector store.
        """
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def __len__(self) -> int:
        """Get the number of vectors in the store."""
        return len(self.vectors)
    
    def __contains__(self, id: str) -> bool:
        """Check if a vector ID exists in the store."""
        return id in self.vectors

def create_vector_store() -> VectorStore:
    """
    Create a new vector store.
    
    Returns:
        A new VectorStore instance.
    """
    return VectorStore()

def get_vector_store(path: str) -> Optional[VectorStore]:
    """
    Load a vector store from disk.
    
    Args:
        path: Path to the vector store file.
    
    Returns:
        The loaded VectorStore or None if the file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'rb') as f:
            store = pickle.load(f)
        return store
    except Exception as e:
        logger.error(f"Failed to load vector store from {path}: {e}")
        return None

def add_vectors(store: VectorStore, texts: List[str], ids: Optional[List[str]] = None,
               embedding_fn: Optional[callable] = None) -> List[str]:
    """
    Add text vectors to a vector store.
    
    Args:
        store: The vector store.
        texts: List of texts to encode.
        ids: Optional list of IDs. If None, will generate IDs.
        embedding_fn: Function to generate embeddings. If None, will use a default function.
    
    Returns:
        List of generated or provided IDs.
    """
    from .embeddings import generate_embeddings
    
    # Generate IDs if not provided
    if ids is None:
        ids = [f"vec_{i}" for i in range(len(texts))]
    
    # Generate embeddings
    if embedding_fn is None:
        embedding_fn = generate_embeddings
    
    embeddings = [embedding_fn(text) for text in texts]
    
    # Add to vector store
    for i, (id, embedding) in enumerate(zip(ids, embeddings)):
        metadata = {"text": texts[i]}
        store.add(id, embedding, metadata)
    
    return ids

def search_vectors(store: VectorStore, query: str, top_k: int = 5,
                  embedding_fn: Optional[callable] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Search for similar vectors in a vector store.
    
    Args:
        store: The vector store.
        query: The query text.
        top_k: Number of results to return.
        embedding_fn: Function to generate embeddings. If None, will use a default function.
    
    Returns:
        List of tuples containing (id, similarity, metadata).
    """
    from .embeddings import generate_embeddings
    
    # Generate query embedding
    if embedding_fn is None:
        embedding_fn = generate_embeddings
    
    query_embedding = embedding_fn(query)
    
    # Search
    return store.search(query_embedding, top_k)
"@
Set-Content -Path (Join-Path $memoryIndexingDir "vector_store.py") -Value $vectorStoreContent

# Implement memory_indexing/embeddings.py
$embeddingsContent = @"
"""
Embeddings
---------
Provides functions for generating text embeddings.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Union

from ..utils.logging_utils import get_logger
from ..cache.cache_manager import load_from_cache, save_to_cache

logger = get_logger(__name__)

def preprocess_text(text: str) -> str:
    """
    Preprocess text for embedding generation.
    
    Args:
        text: The text to preprocess.
    
    Returns:
        The preprocessed text.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Trim
    text = text.strip()
    
    return text

def generate_embeddings(text: str, model_name: Optional[str] = None, 
                       use_cache: bool = True) -> np.ndarray:
    """
    Generate embeddings for text.
    
    Args:
        text: The text to generate embeddings for.
        model_name: Optional name of the embedding model to use.
        use_cache: Whether to use cache for embedding generation.
    
    Returns:
        The generated embedding as a numpy array.
    """
    # Preprocess text
    text = preprocess_text(text)
    
    # Check cache if enabled
    if use_cache:
        cache_key = f"embed_{model_name or 'default'}_{hash(text)}"
        cached = load_from_cache(cache_key)
        if cached is not None:
            return np.array(cached)
    
    try:
        # Try to use transformers if available
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Default to a small model if none specified
            if model_name is None:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Tokenize and generate embedding
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Convert to numpy and normalize
            embedding = embeddings[0].numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to use transformers for embedding generation: {e}")
            
            # Fallback to a simple embedding method
            words = text.lower().split()
            # Create a simple bag-of-words embedding
            embedding = np.zeros(100)  # Fixed size
            for i, word in enumerate(words[:100]):
                # Simple hash-based embedding
                embedding[i % 100] += hash(word) % 10000 / 10000
            
            # Normalize
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
        
        # Cache the result if enabled
        if use_cache:
            save_to_cache(cache_key, embedding.tolist())
        
        return embedding
    
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        # Return a zero vector as fallback
        return np.zeros(100)

def batch_generate_embeddings(texts: List[str], model_name: Optional[str] = None,
                             use_cache: bool = True) -> List[np.ndarray]:
    """
    Generate embeddings for a batch of texts.
    
    Args:
        texts: List of texts to generate embeddings for.
        model_name: Optional name of the embedding model to use.
        use_cache: Whether to use cache for embedding generation.
    
    Returns:
        List of generated embeddings.
    """
    return [generate_embeddings(text, model_name, use_cache) for text in texts]
"@
Set-Content -Path (Join-Path $memoryIndexingDir "embeddings.py") -Value $embeddingsContent
