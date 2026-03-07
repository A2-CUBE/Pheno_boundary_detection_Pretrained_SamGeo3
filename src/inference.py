"""
FTW Model inference utilities.

This module handles:
- Model loading and initialization
- Tiled prediction for large images
- Batch inference across multiple years
"""

import numpy as np
import torch
import os
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm


class FTWInference:
    """Wrapper for Fields of The World segmentation model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize FTW model.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to FTW_v1_3_Class_FULL.ckpt
        device : str, optional
            'cuda' or 'cpu'. Auto-detect if None.
        verbose : bool
            Print loading information
        """
        # Import here to allow module import without torch
        import segmentation_models_pytorch as smp
        
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Build model architecture (must match FTW training)
        self.model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights=None,  # We'll load pre-trained weights
            in_channels=8,         # Bi-temporal Sentinel-2
            classes=3,             # Field, boundary, non-field
        )
        
        # Load weights
        if verbose:
            print(f"Loading FTW model from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (common in PyTorch Lightning)
            state_dict = {
                k.replace('model.', ''): v 
                for k, v in state_dict.items()
            }
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle minor mismatches
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        if verbose:
            print(f"FTW model loaded on {self.device}")
            print(f"  - Encoder: efficientnet-b3")
            print(f"  - Input channels: 8")
            print(f"  - Output classes: 3")
    
    def predict_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Run inference on a single tile.
        
        Parameters
        ----------
        tile : np.ndarray
            Shape (8, H, W), values in [0, 1]
        
        Returns
        -------
        np.ndarray
            Shape (3, H, W) class probabilities
        """
        with torch.no_grad():
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)
            logits = self.model(tile_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs
    
    def predict(
        self,
        input_array: np.ndarray,
        tile_size: int = 256,
        overlap: int = 32,
        show_progress: bool = True
    ) -> Dict:
        """
        Run inference with tiled prediction for large images.
        
        Uses overlapping tiles and averages predictions in overlap regions
        for smoother results.
        
        Parameters
        ----------
        input_array : np.ndarray
            Shape (8, H, W), values in [0, 1]
        tile_size : int
            Size of inference tiles (must be divisible by 32 for U-Net)
        overlap : int
            Overlap between adjacent tiles
        show_progress : bool
            Show progress bar
        
        Returns
        -------
        dict
            'field_prob': Field interior probability map
            'boundary_prob': Field boundary probability map
            'class_map': Argmax class labels (0=non-field, 1=field, 2=boundary)
            'field_mask': Binary field mask (prob > 0.5)
            'raw_probs': Raw 3-channel probability array
        """
        _, H, W = input_array.shape
        
        # Pad to multiple of tile_size
        pad_h = (tile_size - H % tile_size) % tile_size
        pad_w = (tile_size - W % tile_size) % tile_size
        
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(
                input_array,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode='reflect'
            )
        else:
            padded = input_array
        
        _, pH, pW = padded.shape
        
        # Output arrays for accumulation
        output_sum = np.zeros((3, pH, pW), dtype=np.float32)
        count = np.zeros((pH, pW), dtype=np.float32)
        
        stride = tile_size - overlap
        
        # Calculate number of tiles
        n_tiles_y = (pH - tile_size) // stride + 1
        n_tiles_x = (pW - tile_size) // stride + 1
        total_tiles = n_tiles_y * n_tiles_x
        
        # Iterate over tiles
        iterator = range(0, pH - tile_size + 1, stride)
        if show_progress:
            iterator = tqdm(iterator, desc="Inference", total=n_tiles_y)
        
        with torch.no_grad():
            for y in iterator:
                for x in range(0, pW - tile_size + 1, stride):
                    # Extract tile
                    tile = padded[:, y:y+tile_size, x:x+tile_size]
                    
                    # Predict
                    probs = self.predict_tile(tile)
                    
                    # Accumulate
                    output_sum[:, y:y+tile_size, x:x+tile_size] += probs
                    count[y:y+tile_size, x:x+tile_size] += 1
        
        # Average overlapping predictions
        output = output_sum / np.maximum(count, 1)
        
        # Crop to original size
        output = output[:, :H, :W]
        
        # Extract outputs
        # Class indices: 0=Non-field, 1=Field interior, 2=Boundary
        results = {
            'field_prob': output[1],
            'boundary_prob': output[2],
            'nonfield_prob': output[0],
            'class_map': np.argmax(output, axis=0).astype(np.uint8),
            'field_mask': output[1] > 0.5,
            'raw_probs': output,
        }
        
        return results


def download_ftw_model(
    output_path: str = "models/ftw/FTW_v1_3_Class_FULL.ckpt",
    model_name: str = "FTW_v1_3_Class_FULL"
) -> str:
    """
    Download FTW model weights from GitHub releases.
    
    Parameters
    ----------
    output_path : str
        Where to save the model
    model_name : str
        Model variant name
    
    Returns
    -------
    str
        Path to downloaded model
    """
    import urllib.request
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        return output_path
    
    url = f"https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/{model_name}.ckpt"
    
    print(f"Downloading {model_name} from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")
    
    return output_path


def run_multi_year_inference(
    datacube,
    model: FTWInference,
    years: List[int],
    output_dir: str,
    prepare_input_fn=None
) -> Dict:
    """
    Run FTW inference for multiple years.
    
    Parameters
    ----------
    datacube : xr.Dataset
        Cloud-masked multi-temporal datacube
    model : FTWInference
        Loaded FTW model
    years : list
        Years to process
    output_dir : str
        Directory to save results
    prepare_input_fn : callable, optional
        Function to prepare FTW input. Defaults to prepare_ftw_input.
    
    Returns
    -------
    dict
        {year: results_dict} for each year
    """
    from .preprocessing import prepare_ftw_input
    
    if prepare_input_fn is None:
        prepare_input_fn = prepare_ftw_input
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    
    for year in years:
        print(f"\n{'='*50}")
        print(f"Processing year {year}")
        print('='*50)
        
        try:
            # Prepare input
            print("  Preparing bi-temporal composite...")
            input_array = prepare_input_fn(datacube, year)
            print(f"  Input shape: {input_array.shape}")
            print(f"  Value range: [{input_array.min():.3f}, {input_array.max():.3f}]")
            
            # Run inference
            print("  Running inference...")
            results = model.predict(input_array, show_progress=True)
            
            # Statistics
            field_pct = 100 * results['field_mask'].sum() / results['field_mask'].size
            print(f"  Field coverage: {field_pct:.1f}%")
            
            # Save results
            output_file = os.path.join(output_dir, f"ftw_results_{year}.npz")
            np.savez_compressed(
                output_file,
                field_prob=results['field_prob'],
                boundary_prob=results['boundary_prob'],
                class_map=results['class_map'],
                field_mask=results['field_mask'],
            )
            print(f"  Saved to {output_file}")
            
            all_results[year] = results
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[year] = None
    
    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print('='*50)
    successful = sum(1 for r in all_results.values() if r is not None)
    print(f"Successfully processed: {successful}/{len(years)} years")
    
    return all_results


def load_inference_results(output_dir: str, years: List[int]) -> Dict:
    """
    Load previously saved inference results.
    
    Parameters
    ----------
    output_dir : str
        Directory containing .npz files
    years : list
        Years to load
    
    Returns
    -------
    dict
        {year: results_dict} for each year
    """
    results = {}
    
    for year in years:
        filepath = os.path.join(output_dir, f"ftw_results_{year}.npz")
        
        if os.path.exists(filepath):
            data = np.load(filepath)
            results[year] = {
                'field_prob': data['field_prob'],
                'boundary_prob': data['boundary_prob'],
                'class_map': data['class_map'],
                'field_mask': data['field_mask'],
            }
        else:
            print(f"Warning: Results not found for {year}")
            results[year] = None
    
    return results
