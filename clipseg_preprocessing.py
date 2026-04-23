"""
CLIPSeg Data Preprocessing Pipeline
====================================
Converts COCO format datasets (cracks + drywall) into a unified format:
(image, prompt, mask) triplets for text-conditioned segmentation training.

Features:
- Handles polygon segmentation (cracks dataset)
- Converts bounding boxes to masks (drywall dataset)
- Assigns random prompts based on dataset type
- Maintains train/validation split integrity
- Generates metadata CSV files
- Includes visualization utilities
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PROMPT DEFINITIONS
# ============================================================================

CRACK_PROMPTS = [
    "segment crack",
    "segment wall crack",
    "find cracks",
    "highlight crack",
    "detect wall cracks"
]

TAPING_PROMPTS = [
    "segment taping area",
    "segment drywall seam",
    "find drywall joint",
    "highlight joint tape",
    "detect drywall seam"
]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_coco(json_path: str) -> Dict:
    """
    Load COCO format JSON file.
    
    Args:
        json_path: Path to COCO annotations JSON file
        
    Returns:
        Dictionary containing COCO data (images, annotations, categories)
    """
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        logger.info(f"✓ Loaded COCO file: {json_path}")
        logger.info(f"  - Images: {len(coco_data.get('images', []))}")
        logger.info(f"  - Annotations: {len(coco_data.get('annotations', []))}")
        return coco_data
    except Exception as e:
        logger.error(f"✗ Failed to load COCO file: {json_path}")
        logger.error(f"  Error: {e}")
        raise


def create_mask_polygon(annotations: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create binary mask from polygon segmentation annotations.
    
    Args:
        annotations: List of annotation dictionaries (for single image)
        image_shape: Tuple of (height, width)
        
    Returns:
        Binary mask with shape (height, width) where 0=background, 255=object
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for annotation in annotations:
        # Skip if no segmentation data
        if 'segmentation' not in annotation or not annotation['segmentation']:
            continue
        
        # Each annotation can have multiple segmentation polygons
        for segmentation in annotation['segmentation']:
            if len(segmentation) < 6:  # Need at least 3 points (6 coordinates)
                continue
            
            # Convert flat list to polygon points
            polygon = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
            
            # Fill polygon with white (255)
            cv2.fillPoly(mask, [polygon], 255)
    
    return mask


def create_mask_bbox(annotations: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create binary mask from bounding box annotations.
    
    Args:
        annotations: List of annotation dictionaries (for single image)
        image_shape: Tuple of (height, width)
        
    Returns:
        Binary mask with shape (height, width) where 0=background, 255=object
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for annotation in annotations:
        # Skip if no bbox
        if 'bbox' not in annotation or not annotation['bbox']:
            continue
        
        bbox = annotation['bbox']
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Convert to integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Clip to image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image_shape[1] - x)
        h = min(h, image_shape[0] - y)
        
        # Fill rectangle with white (255)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    return mask


def assign_prompt(dataset_type: str) -> str:
    """
    Assign random prompt based on dataset type.
    
    Args:
        dataset_type: Either 'crack' or 'drywall'
        
    Returns:
        Randomly selected prompt string
    """
    if dataset_type == 'crack':
        return random.choice(CRACK_PROMPTS)
    elif dataset_type == 'drywall':
        return random.choice(TAPING_PROMPTS)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def process_dataset(
    dataset_path: str,
    split: str,
    dataset_type: str,
    output_base_path: str,
    segmentation_type: str = 'polygon'
) -> List[Dict]:
    """
    Process a single dataset split.
    
    Args:
        dataset_path: Path to dataset directory
        split: Either 'train' or 'valid'
        dataset_type: Either 'crack' or 'drywall'
        output_base_path: Base path for processed output
        segmentation_type: Either 'polygon' or 'bbox'
        
    Returns:
        List of metadata dictionaries for this split
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {dataset_type.upper()} dataset - {split.upper()} split")
    logger.info(f"{'='*70}")
    
    # Construct paths
    split_dir = os.path.join(dataset_path, split)
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    
    # Images are directly in split directory (not in subdirectory)
    images_dir = split_dir
    
    # Verify paths exist
    if not os.path.exists(images_dir):
        logger.error(f"✗ Split directory not found: {images_dir}")
        return []
    
    if not os.path.exists(json_path):
        logger.error(f"✗ Annotations JSON not found: {json_path}")
        return []
    
    # Load COCO annotations
    coco_data = load_coco(json_path)
    
    # Create output directories
    output_split_path = os.path.join(output_base_path, split)
    output_images_path = os.path.join(output_split_path, 'images')
    output_masks_path = os.path.join(output_split_path, 'masks')
    
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)
    
    logger.info(f"Output paths created:")
    logger.info(f"  - Images: {output_images_path}")
    logger.info(f"  - Masks: {output_masks_path}")
    
    # Build annotation mapping (image_id -> [annotations])
    annotation_map = {}
    for annotation in coco_data.get('annotations', []):
        img_id = annotation['image_id']
        if img_id not in annotation_map:
            annotation_map[img_id] = []
        annotation_map[img_id].append(annotation)
    
    # Process each image
    metadata_list = []
    processed_count = 0
    failed_count = 0
    
    for image_info in coco_data.get('images', []):
        image_id = image_info['id']
        original_filename = image_info['file_name']
        image_height = image_info['height']
        image_width = image_info['width']
        
        # Create renamed filename to avoid collisions
        base_name = os.path.splitext(original_filename)[0]
        renamed_image_name = f"{dataset_type}_{base_name}.jpg"
        renamed_mask_name = f"{dataset_type}_{base_name}.png"
        
        # Source image path
        source_image_path = os.path.join(images_dir, original_filename)
        
        # Check if image exists
        if not os.path.exists(source_image_path):
            logger.warning(f"⚠ Image not found: {source_image_path}")
            failed_count += 1
            continue
        
        try:
            # Read image
            image = cv2.imread(source_image_path)
            if image is None:
                logger.warning(f"⚠ Failed to read image: {source_image_path}")
                failed_count += 1
                continue
            
            # Get actual image dimensions
            actual_height, actual_width = image.shape[:2]
            
            # Generate mask
            image_annotations = annotation_map.get(image_id, [])
            
            if segmentation_type == 'polygon':
                mask = create_mask_polygon(image_annotations, (actual_height, actual_width))
            else:  # bbox
                mask = create_mask_bbox(image_annotations, (actual_height, actual_width))
            
            # Save image
            output_image_path = os.path.join(output_images_path, renamed_image_name)
            cv2.imwrite(output_image_path, image)
            
            # Save mask
            output_mask_path = os.path.join(output_masks_path, renamed_mask_name)
            cv2.imwrite(output_mask_path, mask)
            
            # Assign prompt
            prompt = assign_prompt(dataset_type)
            
            # Create metadata entry
            metadata_entry = {
                'image': renamed_image_name,
                'mask': renamed_mask_name,
                'prompt': prompt,
                'label': dataset_type,
                'original_file': original_filename,
                'height': actual_height,
                'width': actual_width,
                'annotation_count': len(image_annotations)
            }
            metadata_list.append(metadata_entry)
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                logger.info(f"  ✓ Processed {processed_count} images...")
        
        except Exception as e:
            logger.error(f"✗ Error processing {original_filename}: {e}")
            failed_count += 1
            continue
    
    logger.info(f"\nSplit Statistics:")
    logger.info(f"  ✓ Successfully processed: {processed_count}")
    logger.info(f"  ✗ Failed: {failed_count}")
    
    return metadata_list


def save_metadata(metadata_list: List[Dict], output_path: str):
    """
    Save metadata list as CSV file.
    
    Args:
        metadata_list: List of metadata dictionaries
        output_path: Path to save CSV file
    """
    if not metadata_list:
        logger.warning(f"⚠ No metadata to save: {output_path}")
        return
    
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Metadata saved: {output_path}")
    logger.info(f"  - Records: {len(df)}")
    logger.info(f"  - Columns: {list(df.columns)}")


def merge_datasets(
    cracks_dataset_path: str,
    drywall_dataset_path: str,
    output_base_path: str
):
    """
    Main pipeline: merge both datasets while maintaining splits.
    
    Args:
        cracks_dataset_path: Path to cracks dataset
        drywall_dataset_path: Path to drywall dataset
        output_base_path: Base path for output
    """
    logger.info("\n" + "="*70)
    logger.info("CLIPSeg DATA PREPROCESSING PIPELINE")
    logger.info("="*70)
    
    # Create base output directory
    os.makedirs(output_base_path, exist_ok=True)
    
    # Process each split
    all_metadata = {'train': [], 'valid': []}
    
    for split in ['train', 'valid']:
        # Process cracks dataset
        crack_metadata = process_dataset(
            cracks_dataset_path,
            split,
            'crack',
            output_base_path,
            segmentation_type='polygon'
        )
        all_metadata[split].extend(crack_metadata)
        
        # Process drywall dataset
        drywall_metadata = process_dataset(
            drywall_dataset_path,
            split,
            'drywall',
            output_base_path,
            segmentation_type='bbox'
        )
        all_metadata[split].extend(drywall_metadata)
        
        # Save combined metadata for this split
        split_output_path = os.path.join(output_base_path, split)
        metadata_path = os.path.join(split_output_path, 'metadata.csv')
        save_metadata(all_metadata[split], metadata_path)
    
    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("="*70)
    logger.info(f"Train samples: {len(all_metadata['train'])}")
    logger.info(f"Valid samples: {len(all_metadata['valid'])}")
    logger.info(f"Total samples: {len(all_metadata['train']) + len(all_metadata['valid'])}")
    logger.info(f"\nOutput directory: {output_base_path}")
    logger.info("="*70)
    
    return all_metadata


# ============================================================================
# VISUALIZATION UTILITIES (BONUS)
# ============================================================================

def visualize_sample(
    image_path: str,
    mask_path: str,
    prompt: str,
    label: str,
    alpha: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize image with overlaid mask.
    
    Args:
        image_path: Path to image file
        mask_path: Path to mask file
        prompt: Text prompt for this sample
        label: Dataset label (crack or drywall)
        alpha: Transparency of mask overlay (0-1)
        save_path: Optional path to save visualization
        
    Returns:
        Visualization image as numpy array
    """
    # Read image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        logger.error(f"Failed to read image or mask")
        return None
    
    # Resize mask to match image if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create colored mask (green for objects)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 128] = [0, 255, 0]  # Green channel
    
    # Blend image and mask
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    # Add text annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Label: {label} | Prompt: {prompt}"
    cv2.putText(overlay, text, (10, 30), font, 0.6, (255, 255, 255), 2)
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, overlay)
        logger.info(f"✓ Visualization saved: {save_path}")
    
    return overlay


def visualize_batch(
    metadata_csv_path: str,
    dataset_root_path: str,
    num_samples: int = 5,
    output_dir: Optional[str] = None
):
    """
    Visualize multiple samples from processed dataset.
    
    Args:
        metadata_csv_path: Path to metadata.csv file
        dataset_root_path: Root path of processed dataset
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations (optional)
    """
    # Read metadata
    metadata_df = pd.read_csv(metadata_csv_path)
    
    # Random sample
    sample_indices = np.random.choice(len(metadata_df), min(num_samples, len(metadata_df)), replace=False)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for idx in sample_indices:
        row = metadata_df.iloc[idx]
        
        # Construct paths
        split = os.path.basename(os.path.dirname(metadata_csv_path))
        image_path = os.path.join(dataset_root_path, split, 'images', row['image'])
        mask_path = os.path.join(dataset_root_path, split, 'masks', row['mask'])
        
        # Generate visualization
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"sample_{idx}_{row['label']}.jpg")
        
        visualize_sample(
            image_path,
            mask_path,
            row['prompt'],
            row['label'],
            save_path=save_path
        )
        
        logger.info(f"Sample {idx}:")
        logger.info(f"  Image: {row['image']}")
        logger.info(f"  Prompt: {row['prompt']}")
        logger.info(f"  Label: {row['label']}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure paths - UPDATE THESE TO YOUR PATHS
    CRACKS_PATH = "d:/python/Origin/cracks-1"  # Dataset 1: cracks
    DRYWALL_PATH = "d:/python/Origin/Drywall-Join-Detect-1"  # Dataset 2: drywall
    OUTPUT_PATH = "d:/python/Origin/processed"  # Output directory
    
    # Run pipeline
    metadata = merge_datasets(CRACKS_PATH, DRYWALL_PATH, OUTPUT_PATH)
    
    # Optional: Visualize samples
    logger.info("\n" + "="*70)
    logger.info("Generating sample visualizations...")
    logger.info("="*70)
    
    train_metadata_path = os.path.join(OUTPUT_PATH, 'train', 'metadata.csv')
    if os.path.exists(train_metadata_path):
        visualize_batch(
            train_metadata_path,
            OUTPUT_PATH,
            num_samples=3,
            output_dir=os.path.join(OUTPUT_PATH, 'visualizations')
        )
    
    logger.info("\n✓ Pipeline complete!")
