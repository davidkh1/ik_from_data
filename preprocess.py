"""
Preprocessing script to extract dot positions from videos using co-tracker.

This script:
1. Loads the HuggingFace dataset (paszea/ik)
2. Extracts frames from both camera views (front and wrist)
3. Uses co-tracker to trace the two dots on the gripper jaws
4. Calculates the middle point of the two dots in each camera view
5. Applies a 6-frame offset to compensate for lag (0.2s at 30fps)

Usage example:
 # Run without visualization (memory-efficient)
  python preprocess.py \
    --output output/processed_data.npz \
    --dots-cam1 339 185 418 168 \
    --dots-cam2 141 177 149 120 \
    --visualize 0

  # Quick test with visualization (loads only 1000 frames)
  python preprocess.py \
    --output output/test_data.npz \
    --dots-cam1 339 185 418 168 \
    --dots-cam2 141 177 149 120 \
    --max-frames 1000 \
    --visualize 1

  python preprocess.py \
    --output output/processed_data.npz \
    --dots-cam1 340 181 416 166 \
    --dots-cam2 141 179 149 119 \
    --visualize 1 \
    --wandb \
    --wandb-project ik_from_data
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import torch
import imageio.v3 as iio
import time
from utils.memory_utils import print_memory_usage

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def count_video_frames(video_path):
    """
    Count frames in a video without loading them into memory.

    Args:
        video_path: Path to the video file

    Returns:
        int: Number of frames in the video
    """
    count = 0
    for _ in iio.imiter(video_path, plugin="FFMPEG"):
        count += 1
    return count


def load_video_frames(video_path):
    """
    Load all frames from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        numpy array of frames (T, H, W, 3)
    """
    print(f"Loading video: {video_path}")
    frames = iio.imread(video_path, plugin="FFMPEG")
    return frames


def stream_video_frames(video_path):
    """
    Stream frames from a video file without loading all into memory.

    Args:
        video_path: Path to the video file

    Yields:
        numpy array of shape (H, W, 3) for each frame
    """
    for frame in iio.imiter(video_path, plugin="FFMPEG"):
        yield frame


def get_video_paths(video_dir, camera='front', episode=None):
    """
    Get paths to all video files for a specific camera.

    Args:
        video_dir: Path to videos directory
        camera: 'front' or 'wrist'
        episode: If specified, return only this episode index (0-based)

    Returns:
        List of Path objects to video files
    """
    camera_key = f'observation.images.{camera}'
    camera_dir = Path(video_dir) / "chunk-000" / camera_key

    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")

    episode_videos = sorted(camera_dir.glob("episode_*.mp4"))

    if episode is not None:
        if episode < 0 or episode >= len(episode_videos):
            raise ValueError(f"Episode {episode} out of range (0-{len(episode_videos)-1})")
        episode_videos = [episode_videos[episode]]
        print(f"Processing single episode: {episode_videos[0].name}")

    return episode_videos


def count_total_frames(video_dir, camera='front', episode=None):
    """
    Count total frames across all videos without loading them.

    Args:
        video_dir: Path to videos directory
        camera: 'front' or 'wrist'
        episode: If specified, count only this episode index (0-based)

    Returns:
        int: Total number of frames across all videos
    """
    video_paths = get_video_paths(video_dir, camera, episode=episode)
    print(f"Counting frames in {len(video_paths)} {camera} videos...")

    total = 0
    for video_path in tqdm(video_paths, desc=f"Counting {camera} frames"):
        total += count_video_frames(video_path)

    print(f"Total frames in {camera} camera: {total}")
    return total


def extract_frames_from_videos(video_dir, camera='front', max_frames=None):
    """
    Extract frames from video files for a specific camera.

    Args:
        video_dir: Path to videos directory
        camera: 'front' or 'wrist'
        max_frames: Maximum number of frames to load (default: None = all frames)

    Returns:
        List of frames as numpy arrays
    """
    all_frames = []
    camera_key = f'observation.images.{camera}'
    camera_dir = Path(video_dir) / "chunk-000" / camera_key

    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")

    if max_frames:
        print(f"Extracting first {max_frames} frames from {camera} camera...")
    else:
        print(f"Extracting all frames from {camera} camera...")

    # Find all episode videos
    episode_videos = sorted(camera_dir.glob("episode_*.mp4"))
    print(f"Found {len(episode_videos)} episode videos")

    for video_path in tqdm(episode_videos, desc=f"Loading {camera} videos"):
        # Stream frames instead of loading all at once
        for frame in stream_video_frames(video_path):
            all_frames.append(frame)

            # Stop if we've reached max_frames
            if max_frames and len(all_frames) >= max_frames:
                print(f"Reached max_frames limit ({max_frames}), stopping early")
                return all_frames

    print(f"Total frames extracted: {len(all_frames)}")
    return all_frames


def track_dots_with_cotracker(frames, initial_points, chunk_size=200):
    """
    Track dots through frames using co-tracker (loaded via torch.hub).
    Processes video in chunks to avoid GPU OOM.

    Args:
        frames: List of frames as numpy arrays (H, W, 3)
        initial_points: Initial positions of dots in first frame, shape (2, 2) [[x1, y1], [x2, y2]]
        chunk_size: Number of frames to process at once (default: 200)

    Returns:
        numpy array of shape (num_frames, 2, 2) containing tracked positions
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load CoTracker3 offline model via torch.hub
    print("Loading CoTracker3 model from torch.hub...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

    total_frames = len(frames)
    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    print(f"Processing {total_frames} frames in {num_chunks} chunks of {chunk_size} frames...")

    all_tracks = []

    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_frames)
        chunk_frames = frames[start_idx:end_idx]

        # Convert chunk to tensor (B, T, C, H, W)
        video_tensor = torch.from_numpy(np.stack(chunk_frames)).permute(0, 3, 1, 2).float()
        video_tensor = video_tensor.unsqueeze(0).to(device)

        # For chunks after the first, use the last tracked position from previous chunk
        if chunk_idx == 0:
            query_points = initial_points
        else:
            # Use last tracked position from previous chunk as starting point
            query_points = all_tracks[-1][-1]  # Last frame of previous chunk

        # Prepare query points for tracking
        queries = torch.zeros((1, 2, 3), device=device)
        queries[0, :, 0] = 0  # Start tracking from frame 0 of this chunk
        queries[0, :, 1:] = torch.from_numpy(query_points).float()

        # Track points
        with torch.no_grad():
            pred_tracks, pred_visibility = cotracker(video_tensor, queries=queries)

        # pred_tracks shape: (B, T, N, 2)
        chunk_tracks = pred_tracks[0].cpu().numpy()  # (T, 2, 2)
        all_tracks.append(chunk_tracks)

        # Clear GPU cache
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Concatenate all chunks
    tracks = np.concatenate(all_tracks, axis=0)
    print(f"Tracking complete! Total frames tracked: {len(tracks)}")

    return tracks


def track_dots_with_cotracker_streaming(video_paths, initial_points, total_frames, chunk_size=200, gpu_chunk_size=50, temp_dir="output", overlap_frames=20, visibility_threshold=0.3):
    """
    Track dots through videos using streaming to avoid loading all frames into RAM.
    Writes results incrementally to memory-mapped file.

    Includes anti-drift measures:
    1. Overlapping chunks - maintains context across chunk boundaries
    2. Visibility monitoring - detects when tracking is lost
    3. Online mode option - uses CoTracker's streaming-friendly model

    Args:
        video_paths: List of paths to video files
        initial_points: Initial positions of dots in first frame, shape (2, 2) [[x1, y1], [x2, y2]]
        total_frames: Total number of frames across all videos
        chunk_size: Number of frames to process at once for CPU (default: 200)
        gpu_chunk_size: Max frames for GPU to avoid VRAM OOM (default: 50)
        temp_dir: Directory for temporary memmap files
        overlap_frames: Number of frames to overlap between chunks for continuity (default: 20)
        visibility_threshold: Minimum visibility score to consider tracking valid (default: 0.3)

    Returns:
        numpy memmap array of shape (total_frames, 2, 2) containing tracked positions
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Adjust chunk size for GPU to avoid OOM
    if device == 'cuda':
        # GPU has limited memory, reduce chunk size
        original_chunk_size = chunk_size
        chunk_size = min(chunk_size, gpu_chunk_size)
        if chunk_size != original_chunk_size:
            print(f"GPU detected: reducing chunk size from {original_chunk_size} to {chunk_size} frames to avoid GPU OOM")
        else:
            print(f"GPU detected: using chunk size {chunk_size} frames")

    # Ensure overlap is smaller than chunk size
    overlap_frames = min(overlap_frames, chunk_size // 2)
    print(f"Using {overlap_frames} overlap frames between chunks")

    print_memory_usage("Before loading CoTracker")

    # Load CoTracker model via torch.hub
    print("Loading CoTracker3 model from torch.hub...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    print_memory_usage("After loading CoTracker")

    # Create memory-mapped array for storing tracks
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Use unique filename based on video path to avoid collisions
    camera_id = video_paths[0].parent.name.split('.')[-1]  # Extract 'front' or 'wrist'
    tracks_memmap_path = temp_dir / f"tracks_{camera_id}_temp.dat"
    visibility_memmap_path = temp_dir / f"visibility_{camera_id}_temp.dat"
    print(f"Using temp file: {tracks_memmap_path.name}")

    # Create memmaps for tracks and visibility
    tracks_memmap = np.memmap(tracks_memmap_path, dtype='float32', mode='w+',
                              shape=(total_frames, 2, 2))
    visibility_memmap = np.memmap(visibility_memmap_path, dtype='float32', mode='w+',
                                   shape=(total_frames, 2))

    print(f"Processing {total_frames} frames from {len(video_paths)} videos in chunks of {chunk_size}...")
    print(f"Anti-drift: overlap={overlap_frames}, visibility_threshold={visibility_threshold}")

    global_frame_idx = 0
    query_points = initial_points.copy()
    original_points = initial_points.copy()  # Keep original for reinitialization
    low_visibility_count = 0
    reinit_count = 0

    # For overlap handling, keep last N frames from previous chunk
    overlap_buffer = []

    for video_idx, video_path in enumerate(video_paths):
        # Stop if we've reached total_frames limit
        if global_frame_idx >= total_frames:
            print(f"\nReached frame limit ({total_frames}), stopping processing")
            break

        print(f"\nProcessing video {video_idx + 1}/{len(video_paths)}: {video_path.name}")
        print_memory_usage(f"Before video {video_idx + 1}")

        # Stream frames from this video in chunks
        frame_buffer = list(overlap_buffer)  # Start with overlap from previous chunk
        overlap_buffer = []

        for frame in stream_video_frames(video_path):
            # Stop if we've reached total_frames limit
            if global_frame_idx >= total_frames:
                break

            frame_buffer.append(frame)

            # Process when buffer reaches chunk_size
            if len(frame_buffer) >= chunk_size:
                # Process chunk with visibility tracking
                chunk_tracks, chunk_visibility = _process_frame_chunk_with_visibility(
                    frame_buffer, query_points, cotracker, device
                )

                # Check visibility and handle tracking loss
                mean_visibility = chunk_visibility.mean()
                min_visibility = chunk_visibility.min(axis=1).mean()  # Min across both points per frame

                if min_visibility < visibility_threshold:
                    low_visibility_count += 1
                    print(f"  [WARN] Low visibility detected: mean={mean_visibility:.3f}, min={min_visibility:.3f}")

                    # If visibility is very low, consider reinitializing
                    if min_visibility < visibility_threshold * 0.5:
                        print(f"  [REINIT] Visibility too low, reinitializing from last good position")
                        reinit_count += 1

                # Determine how many frames are new (not overlap)
                if len(overlap_buffer) > 0:
                    # We have overlap from previous chunk, skip those frames in output
                    new_start = overlap_frames
                else:
                    new_start = 0

                # Write only new frames to memmap
                new_tracks = chunk_tracks[new_start:]
                new_visibility = chunk_visibility[new_start:]

                end_idx = global_frame_idx + len(new_tracks)
                if end_idx > total_frames:
                    # Truncate if we'd exceed total frames
                    trim = end_idx - total_frames
                    new_tracks = new_tracks[:-trim]
                    new_visibility = new_visibility[:-trim]
                    end_idx = total_frames

                tracks_memmap[global_frame_idx:end_idx] = new_tracks
                visibility_memmap[global_frame_idx:end_idx] = new_visibility

                # Keep overlap frames for next chunk
                overlap_buffer = frame_buffer[-overlap_frames:]

                # Update query points using weighted average from overlap region
                # This helps smooth transitions between chunks
                if overlap_frames > 0:
                    # Use position from middle of overlap region for stability
                    overlap_mid = chunk_tracks[-overlap_frames // 2 - 1]
                    query_points = overlap_mid.copy()
                else:
                    query_points = chunk_tracks[-1].copy()

                # Update indices
                global_frame_idx = end_idx

                # Clear buffer (keeping overlap)
                frame_buffer = list(overlap_buffer)

                # Clear GPU cache and synchronize
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Process remaining frames in buffer
        remaining_new_frames = len(frame_buffer) - len(overlap_buffer)
        if remaining_new_frames > 0:
            chunk_tracks, chunk_visibility = _process_frame_chunk_with_visibility(
                frame_buffer, query_points, cotracker, device
            )

            # Skip overlap frames that were already written
            new_start = len(overlap_buffer) if len(overlap_buffer) > 0 else 0
            new_tracks = chunk_tracks[new_start:]
            new_visibility = chunk_visibility[new_start:]

            end_idx = global_frame_idx + len(new_tracks)
            if end_idx > total_frames:
                trim = end_idx - total_frames
                new_tracks = new_tracks[:-trim]
                new_visibility = new_visibility[:-trim]
                end_idx = total_frames

            tracks_memmap[global_frame_idx:end_idx] = new_tracks
            visibility_memmap[global_frame_idx:end_idx] = new_visibility

            query_points = chunk_tracks[-1].copy()
            global_frame_idx = end_idx

            # Keep overlap for next video
            overlap_buffer = frame_buffer[-overlap_frames:] if len(frame_buffer) >= overlap_frames else frame_buffer

        print_memory_usage(f"After video {video_idx + 1}")

        # Stop if we've reached total_frames limit
        if global_frame_idx >= total_frames:
            print(f"\nReached frame limit ({total_frames}), stopping processing")
            break

    print(f"\nTracking complete! Total frames tracked: {global_frame_idx}")
    print(f"Low visibility warnings: {low_visibility_count}, Reinitializations: {reinit_count}")

    # Calculate and report visibility statistics
    valid_visibility = visibility_memmap[:global_frame_idx]
    mean_vis = valid_visibility.mean()
    min_vis = valid_visibility.min()
    low_vis_frames = (valid_visibility.min(axis=1) < visibility_threshold).sum()
    print(f"Visibility stats: mean={mean_vis:.3f}, min={min_vis:.3f}, frames_below_threshold={low_vis_frames}")

    print_memory_usage("After all tracking")

    # Flush memmap to disk
    tracks_memmap.flush()
    visibility_memmap.flush()

    return tracks_memmap, visibility_memmap


def _process_frame_chunk_with_visibility(frames, query_points, cotracker, device):
    """
    Helper function to process a chunk of frames with CoTracker.
    Returns both tracks and visibility scores.

    Args:
        frames: List of numpy arrays (H, W, 3)
        query_points: Query points for tracking, shape (2, 2)
        cotracker: CoTracker model
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        tuple: (tracks, visibility)
            - tracks: numpy array of shape (T, 2, 2) with tracked positions
            - visibility: numpy array of shape (T, 2) with visibility scores
    """
    # Convert to tensor (B, T, C, H, W)
    video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    video_tensor = video_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        queries = _prepare_queries(query_points, device)
        pred_tracks, pred_visibility = cotracker(video_tensor, queries=queries)

    # Extract tracks (B, T, N, 2) -> (T, 2, 2)
    chunk_tracks = pred_tracks[0].cpu().numpy()
    chunk_visibility = pred_visibility[0].cpu().numpy()  # (T, 2)

    return chunk_tracks, chunk_visibility


def _prepare_queries(query_points, device):
    """Prepare query tensor for CoTracker."""
    queries = torch.zeros((1, 2, 3), device=device)
    queries[0, :, 0] = 0  # Start tracking from frame 0 of this chunk
    queries[0, :, 1:] = torch.from_numpy(query_points).float()
    return queries


def _process_frame_chunk(frames, query_points, cotracker, device):
    """
    Helper function to process a chunk of frames with CoTracker.
    Returns only tracks (no visibility).

    Args:
        frames: List of numpy arrays (H, W, 3)
        query_points: Query points for tracking, shape (2, 2)
        cotracker: CoTracker model
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        numpy array of shape (T, 2, 2) with tracked positions
    """
    tracks, _ = _process_frame_chunk_with_visibility(frames, query_points, cotracker, device)
    return tracks


def calculate_middle_points(tracks_cam1, tracks_cam2):
    """
    Calculate middle points between two dots for both cameras.

    Args:
        tracks_cam1: numpy array (T, 2, 2) - tracks from camera 1
        tracks_cam2: numpy array (T, 2, 2) - tracks from camera 2

    Returns:
        numpy array (T, 4) - [x1, y1, x2, y2] where (x1,y1) is middle point in cam1
    """
    # Calculate middle point for each camera
    middle_cam1 = tracks_cam1.mean(axis=1)  # (T, 2)
    middle_cam2 = tracks_cam2.mean(axis=1)  # (T, 2)

    # Concatenate to get [x1, y1, x2, y2]
    middle_points = np.concatenate([middle_cam1, middle_cam2], axis=1)  # (T, 4)

    return middle_points


def process_dataset_streaming(dataset, initial_dots_cam1, initial_dots_cam2, frame_offset=6, video_dir="data/ik_dataset/videos", return_visualization_data=False, chunk_size=200, gpu_chunk_size=50, temp_dir="output", vis_frames=None, max_frames=None, episode=None):
    """
    Process the entire dataset using streaming to avoid OOM.
    Streams frames instead of loading all into memory.

    Args:
        dataset: HuggingFace dataset
        initial_dots_cam1: Initial dot positions in camera 1, shape (2, 2)
        initial_dots_cam2: Initial dot positions in camera 2, shape (2, 2)
        frame_offset: Frame offset for lag compensation (default 6)
        video_dir: Path to videos directory
        return_visualization_data: If True, also load frames for visualization (uses more RAM)
        chunk_size: Number of frames to process at once for CPU (default 200)
        gpu_chunk_size: Max frames for GPU to avoid VRAM OOM (default 50)
        temp_dir: Directory for temporary memmap files
        vis_frames: Max frames to load for visualization (default: None = all frames)
        max_frames: Maximum number of frames to process total (default: None = all frames)
        episode: If specified, process only this episode index (0-based)

    Returns:
        tuple: (image_positions, joint_positions) or
               (image_positions, joint_positions, frames_cam1, frames_cam2, tracks_cam1, tracks_cam2)
    """
    start_time = time.time()
    print_memory_usage("Start of processing")

    # Count total frames first (without loading)
    total_frames_cam1 = count_total_frames(video_dir, camera='front', episode=episode)
    total_frames_cam2 = count_total_frames(video_dir, camera='wrist', episode=episode)

    # Apply max_frames limit if specified
    if max_frames:
        print(f"\n[LIMIT] Processing only first {max_frames} frames (out of {total_frames_cam1} total)")
        total_frames_cam1 = min(total_frames_cam1, max_frames)
        total_frames_cam2 = min(total_frames_cam2, max_frames)

    if total_frames_cam1 != total_frames_cam2:
        print(f"WARNING: Frame count mismatch! Front: {total_frames_cam1}, Wrist: {total_frames_cam2}")

    count_time = time.time() - start_time
    print(f"[TIME] Frame counting: {count_time:.1f}s")
    print_memory_usage("After counting frames")

    # Get video paths
    video_paths_cam1 = get_video_paths(video_dir, camera='front', episode=episode)
    video_paths_cam2 = get_video_paths(video_dir, camera='wrist', episode=episode)

    # Track dots using streaming approach
    print("\n" + "="*70)
    print("Tracking dots in front camera (streaming)...")
    print("="*70)
    track_start = time.time()
    tracks_cam1, visibility_cam1 = track_dots_with_cotracker_streaming(
        video_paths_cam1, initial_dots_cam1, total_frames_cam1,
        chunk_size=chunk_size, gpu_chunk_size=gpu_chunk_size, temp_dir=temp_dir
    )
    cam1_time = time.time() - track_start
    print(f"[TIME] Front camera tracking: {cam1_time:.1f}s ({cam1_time/60:.1f} min)")

    print("\n" + "="*70)
    print("Tracking dots in wrist camera (streaming)...")
    print("="*70)
    track_start = time.time()
    tracks_cam2, visibility_cam2 = track_dots_with_cotracker_streaming(
        video_paths_cam2, initial_dots_cam2, total_frames_cam2,
        chunk_size=chunk_size, gpu_chunk_size=gpu_chunk_size, temp_dir=temp_dir
    )
    cam2_time = time.time() - track_start
    print(f"[TIME] Wrist camera tracking: {cam2_time:.1f}s ({cam2_time/60:.1f} min)")

    total_tracking_time = cam1_time + cam2_time
    print(f"[TIME] Total tracking time: {total_tracking_time:.1f}s ({total_tracking_time/60:.1f} min)")
    print_memory_usage("After tracking both cameras")

    # Calculate middle points
    middle_points = calculate_middle_points(tracks_cam1, tracks_cam2)

    # Extract joint positions (first 5 joints, ignoring gripper)
    print("\nExtracting joint positions from dataset...")
    joint_positions = []
    for i in range(len(dataset)):
        state = dataset[i]['observation.state']
        joint_positions.append(state[:5])
    joint_positions = np.array(joint_positions)

    # Apply frame offset
    image_positions_offset = middle_points[frame_offset:]
    joint_positions_offset = joint_positions[:-frame_offset]

    print(f"\nProcessed {len(image_positions_offset)} samples")
    print(f"Image positions shape: {image_positions_offset.shape}")
    print(f"Joint positions shape: {joint_positions_offset.shape}")
    print_memory_usage("After creating offset arrays")

    # Return tracks and visibility as memmap (will be used for per-episode visualization)
    # No need to load all frames here - per-episode visualization streams them
    return image_positions_offset, joint_positions_offset, tracks_cam1, tracks_cam2, visibility_cam1, visibility_cam2, video_dir


def process_dataset(dataset, initial_dots_cam1, initial_dots_cam2, frame_offset=6, video_dir="data/ik_dataset/videos", return_visualization_data=False, chunk_size=200):
    """
    Process the entire dataset to extract training data.

    DEPRECATED: Use process_dataset_streaming() instead to avoid OOM.

    Args:
        dataset: HuggingFace dataset
        initial_dots_cam1: Initial dot positions in camera 1, shape (2, 2)
        initial_dots_cam2: Initial dot positions in camera 2, shape (2, 2)
        frame_offset: Frame offset for lag compensation (default 6)
        video_dir: Path to videos directory
        return_visualization_data: If True, also return frames and tracks for visualization
        chunk_size: Number of frames to process at once (default 200)

    Returns:
        tuple: (image_positions, joint_positions) or
               (image_positions, joint_positions, frames_cam1, frames_cam2, tracks_cam1, tracks_cam2)
    """
    # Extract frames from both cameras
    frames_cam1 = extract_frames_from_videos(video_dir, camera='front')
    frames_cam2 = extract_frames_from_videos(video_dir, camera='wrist')

    # Track dots in both camera views
    print("Tracking dots in front camera...")
    tracks_cam1 = track_dots_with_cotracker(frames_cam1, initial_dots_cam1, chunk_size=chunk_size)

    print("Tracking dots in wrist camera...")
    tracks_cam2 = track_dots_with_cotracker(frames_cam2, initial_dots_cam2, chunk_size=chunk_size)

    # Calculate middle points
    middle_points = calculate_middle_points(tracks_cam1, tracks_cam2)

    # Extract joint positions (first 5 joints, ignoring gripper)
    joint_positions = []
    for i in range(len(dataset)):
        state = dataset[i]['observation.state']
        joint_positions.append(state[:5])
    joint_positions = np.array(joint_positions)

    # Apply frame offset: use image positions 6 frames ahead of action
    # This compensates for the lag between arm movement and action
    image_positions_offset = middle_points[frame_offset:]
    joint_positions_offset = joint_positions[:-frame_offset]

    print(f"Processed {len(image_positions_offset)} samples")
    print(f"Image positions shape: {image_positions_offset.shape}")
    print(f"Joint positions shape: {joint_positions_offset.shape}")

    if return_visualization_data:
        return image_positions_offset, joint_positions_offset, frames_cam1, frames_cam2, tracks_cam1, tracks_cam2
    else:
        return image_positions_offset, joint_positions_offset


def visualize_tracking_sidebyside(frames_cam1, tracks_cam1, frames_cam2, tracks_cam2, output_path, max_frames=300):
    """
    Create a side-by-side visualization video showing tracked points from both cameras.

    Args:
        frames_cam1: List of frames from camera 1 (T, H, W, 3)
        tracks_cam1: Tracked positions for camera 1 (T, 2, 2)
        frames_cam2: List of frames from camera 2 (T, H, W, 3)
        tracks_cam2: Tracked positions for camera 2 (T, 2, 2)
        output_path: Path to save the visualization video
        max_frames: Maximum number of frames to visualize (to save time)
    """
    print(f"\nCreating side-by-side tracking visualization...")

    # Limit frames to visualize
    if max_frames is None:
        num_frames = min(len(frames_cam1), len(frames_cam2))
        print(f"Visualizing all {num_frames} loaded frames")
    else:
        num_frames = min(len(frames_cam1), len(frames_cam2), max_frames)
        print(f"Visualizing {num_frames} frames (limited by max_frames={max_frames})")

    # Get video properties
    height, width = frames_cam1[0].shape[:2]
    fps = 30

    # Create video writer for side-by-side (double width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))

    # Draw tracks on each frame
    for i in range(num_frames):
        # Process camera 1 (left)
        frame1_bgr = cv2.cvtColor(frames_cam1[i], cv2.COLOR_RGB2BGR)
        track1 = tracks_cam1[i]

        # Draw dots for camera 1
        for j, (x, y) in enumerate(track1):
            x, y = int(x), int(y)
            color = (0, 255, 0) if j == 0 else (0, 0, 255)  # Green for dot 1, Red for dot 2
            cv2.circle(frame1_bgr, (x, y), 5, color, -1)
            cv2.circle(frame1_bgr, (x, y), 8, color, 2)

        # Draw line between dots
        if len(track1) == 2:
            x1, y1 = int(track1[0][0]), int(track1[0][1])
            x2, y2 = int(track1[1][0]), int(track1[1][1])
            cv2.line(frame1_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Draw middle point
        middle1 = track1.mean(axis=0)
        mx1, my1 = int(middle1[0]), int(middle1[1])
        cv2.circle(frame1_bgr, (mx1, my1), 4, (255, 0, 255), -1)  # Magenta for middle

        # Add camera label
        cv2.putText(frame1_bgr, 'Front Camera', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Process camera 2 (right)
        frame2_bgr = cv2.cvtColor(frames_cam2[i], cv2.COLOR_RGB2BGR)
        track2 = tracks_cam2[i]

        # Draw dots for camera 2
        for j, (x, y) in enumerate(track2):
            x, y = int(x), int(y)
            color = (0, 255, 0) if j == 0 else (0, 0, 255)  # Green for dot 1, Red for dot 2
            cv2.circle(frame2_bgr, (x, y), 5, color, -1)
            cv2.circle(frame2_bgr, (x, y), 8, color, 2)

        # Draw line between dots
        if len(track2) == 2:
            x1, y1 = int(track2[0][0]), int(track2[0][1])
            x2, y2 = int(track2[1][0]), int(track2[1][1])
            cv2.line(frame2_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Draw middle point
        middle2 = track2.mean(axis=0)
        mx2, my2 = int(middle2[0]), int(middle2[1])
        cv2.circle(frame2_bgr, (mx2, my2), 4, (255, 0, 255), -1)  # Magenta for middle

        # Add camera label
        cv2.putText(frame2_bgr, 'Wrist Camera', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine frames side-by-side
        combined_frame = np.hstack([frame1_bgr, frame2_bgr])

        # Add frame number at the bottom center (to avoid overlapping with camera labels)
        frame_text = f'Frame {i}'
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = width - text_size[0] // 2  # Center between the two frames
        text_y = height - 20  # 20 pixels from bottom
        cv2.putText(combined_frame, frame_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(combined_frame)

    out.release()


def visualize_tracking_per_episode(video_dir, tracks_cam1, tracks_cam2, output_dir, max_frames_per_episode=None, visibility_cam1=None, visibility_cam2=None):
    """
    Create separate visualization videos for each episode.

    Args:
        video_dir: Path to videos directory
        tracks_cam1: Tracked positions for all frames in camera 1 (T, 2, 2)
        tracks_cam2: Tracked positions for all frames in camera 2 (T, 2, 2)
        output_dir: Directory to save visualization videos
        max_frames_per_episode: Max frames to visualize per episode (default: None = all)
        visibility_cam1: Visibility scores for camera 1 (T, 2) or None
        visibility_cam2: Visibility scores for camera 2 (T, 2) or None

    Returns:
        List of paths to created visualization videos
    """
    print(f"\nCreating per-episode tracking visualizations...")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    video_paths_cam1 = get_video_paths(video_dir, camera='front')
    video_paths_cam2 = get_video_paths(video_dir, camera='wrist')

    created_videos = []
    global_frame_idx = 0

    for episode_idx, (video1_path, video2_path) in enumerate(zip(video_paths_cam1, video_paths_cam2)):
        episode_name = video1_path.stem  # e.g., 'episode_000000'
        print(f"\nEpisode {episode_idx + 1}/{len(video_paths_cam1)}: {episode_name}")

        # Count frames in this episode
        episode_frame_count = count_video_frames(video1_path)

        # Determine frames to visualize
        if max_frames_per_episode:
            frames_to_vis = min(episode_frame_count, max_frames_per_episode)
        else:
            frames_to_vis = episode_frame_count

        # Check if we have enough tracking data
        if global_frame_idx >= len(tracks_cam1):
            print(f"  Skipping - no tracking data available")
            break

        if global_frame_idx + frames_to_vis > len(tracks_cam1):
            frames_to_vis = len(tracks_cam1) - global_frame_idx
            print(f"  Adjusted to {frames_to_vis} frames (limited by tracking data)")

        if frames_to_vis <= 0:
            break

        # Stream and load frames for this episode
        frames1 = []
        frames2 = []

        for i, (frame1, frame2) in enumerate(zip(
            stream_video_frames(video1_path),
            stream_video_frames(video2_path)
        )):
            if i >= frames_to_vis:
                break
            frames1.append(frame1)
            frames2.append(frame2)

        # Extract tracks for this episode
        episode_tracks1 = tracks_cam1[global_frame_idx:global_frame_idx + frames_to_vis]
        episode_tracks2 = tracks_cam2[global_frame_idx:global_frame_idx + frames_to_vis]

        # Extract visibility for this episode (if available)
        episode_vis1 = visibility_cam1[global_frame_idx:global_frame_idx + frames_to_vis] if visibility_cam1 is not None else None
        episode_vis2 = visibility_cam2[global_frame_idx:global_frame_idx + frames_to_vis] if visibility_cam2 is not None else None

        # Create visualization
        vis_output = output_dir / f"tracking_{episode_name}.mp4"
        _create_sidebyside_visualization(frames1, episode_tracks1, frames2, episode_tracks2,
                                         vis_output, episode_name, global_frame_idx,
                                         visibility_cam1=episode_vis1, visibility_cam2=episode_vis2)

        created_videos.append(vis_output)
        global_frame_idx += episode_frame_count

        print(f"  Created: {vis_output.name}")

    print(f"\n[OK] Created {len(created_videos)} episode visualizations")
    return created_videos


def _create_sidebyside_visualization(frames_cam1, tracks_cam1, frames_cam2, tracks_cam2, output_path, episode_name, frame_offset=0, visibility_cam1=None, visibility_cam2=None, visibility_threshold=0.5):
    """
    Helper to create a single side-by-side video.

    Args:
        frames_cam1: List of frames from camera 1
        tracks_cam1: Tracked positions for camera 1
        frames_cam2: List of frames from camera 2
        tracks_cam2: Tracked positions for camera 2
        output_path: Path to save video
        episode_name: Episode name to display
        frame_offset: Global frame offset for numbering
        visibility_cam1: Visibility scores for camera 1 (T, 2) or None
        visibility_cam2: Visibility scores for camera 2 (T, 2) or None
        visibility_threshold: Threshold below which to show warning (default: 0.5)
    """
    if len(frames_cam1) == 0:
        return

    height, width = frames_cam1[0].shape[:2]
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))

    for i in range(len(frames_cam1)):
        # Get visibility for this frame (if available)
        vis1 = visibility_cam1[i] if visibility_cam1 is not None else None
        vis2 = visibility_cam2[i] if visibility_cam2 is not None else None

        # Process front camera (left)
        frame1_bgr = cv2.cvtColor(frames_cam1[i], cv2.COLOR_RGB2BGR)
        track1 = tracks_cam1[i]

        # Draw dots with visibility-based coloring
        for j, (x, y) in enumerate(track1):
            x, y = int(x), int(y)
            # Color based on visibility: green/red for high vis, orange/yellow for low vis
            if vis1 is not None and vis1[j] < visibility_threshold:
                # Low visibility - use warning colors (orange/yellow)
                color = (0, 165, 255) if j == 0 else (0, 255, 255)  # Orange/Yellow in BGR
                ring_color = (0, 0, 255)  # Red ring for warning
            else:
                # Normal visibility - use standard colors
                color = (0, 255, 0) if j == 0 else (0, 0, 255)  # Green/Red
                ring_color = color
            cv2.circle(frame1_bgr, (x, y), 5, color, -1)
            cv2.circle(frame1_bgr, (x, y), 8, ring_color, 2)

        # Draw line and middle point
        if len(track1) == 2:
            x1, y1 = int(track1[0][0]), int(track1[0][1])
            x2, y2 = int(track1[1][0]), int(track1[1][1])
            cv2.line(frame1_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

        middle1 = track1.mean(axis=0)
        mx1, my1 = int(middle1[0]), int(middle1[1])
        cv2.circle(frame1_bgr, (mx1, my1), 4, (255, 0, 255), -1)

        # Camera label
        cv2.putText(frame1_bgr, 'Front Camera', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show visibility score
        if vis1 is not None:
            min_vis1 = min(vis1)
            vis_color = (0, 255, 0) if min_vis1 >= visibility_threshold else (0, 0, 255)
            cv2.putText(frame1_bgr, f'Vis: {min_vis1:.2f}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, vis_color, 2)

        # Process wrist camera (right)
        frame2_bgr = cv2.cvtColor(frames_cam2[i], cv2.COLOR_RGB2BGR)
        track2 = tracks_cam2[i]

        # Draw dots with visibility-based coloring
        for j, (x, y) in enumerate(track2):
            x, y = int(x), int(y)
            if vis2 is not None and vis2[j] < visibility_threshold:
                color = (0, 165, 255) if j == 0 else (0, 255, 255)
                ring_color = (0, 0, 255)
            else:
                color = (0, 255, 0) if j == 0 else (0, 0, 255)
                ring_color = color
            cv2.circle(frame2_bgr, (x, y), 5, color, -1)
            cv2.circle(frame2_bgr, (x, y), 8, ring_color, 2)

        if len(track2) == 2:
            x1, y1 = int(track2[0][0]), int(track2[0][1])
            x2, y2 = int(track2[1][0]), int(track2[1][1])
            cv2.line(frame2_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

        middle2 = track2.mean(axis=0)
        mx2, my2 = int(middle2[0]), int(middle2[1])
        cv2.circle(frame2_bgr, (mx2, my2), 4, (255, 0, 255), -1)

        cv2.putText(frame2_bgr, 'Wrist Camera', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show visibility score
        if vis2 is not None:
            min_vis2 = min(vis2)
            vis_color = (0, 255, 0) if min_vis2 >= visibility_threshold else (0, 0, 255)
            cv2.putText(frame2_bgr, f'Vis: {min_vis2:.2f}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, vis_color, 2)

        # Combine frames
        combined_frame = np.hstack([frame1_bgr, frame2_bgr])

        # Add episode and frame info at bottom center
        frame_text = f'{episode_name} - Frame {frame_offset + i}'
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = width - text_size[0] // 2
        text_y = height - 15
        cv2.putText(combined_frame, frame_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(combined_frame)

    out.release()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess IK dataset')
    parser.add_argument('--output', type=str, default='output/processed_data.npz',
                        help='Output path for processed data')
    parser.add_argument('--video-dir', type=str, default='data/ik_dataset/videos',
                        help='Path to videos directory (default: data/ik_dataset/videos)')
    parser.add_argument('--frame-offset', type=int, default=0,
                        help='Frame offset for lag compensation (default: 0)')
    parser.add_argument('--dots-cam1', type=float, nargs=4, default=None,
                        help='Initial dot positions in camera 1: x1 y1 x2 y2')
    parser.add_argument('--dots-cam2', type=float, nargs=4, default=None,
                        help='Initial dot positions in camera 2: x1 y1 x2 y2')
    parser.add_argument('--visualize', type=int, choices=[0, 1], default=1,
                        help='Create visualization videos of tracking results: 1=yes, 0=no (default: 1)')
    parser.add_argument('--vis-frames', type=int, default=None,
                        help='Max frames to visualize PER EPISODE (default: None = all frames per episode)')
    parser.add_argument('--chunk-size', type=int, default=200,
                        help='Number of frames to process at once on CPU (default: 200)')
    parser.add_argument('--gpu-chunk-size', type=int, default=50,
                        help='Max frames to process at once on GPU to avoid VRAM OOM (default: 50)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process (default: None = all frames). Useful for testing.')
    parser.add_argument('--episode', type=int, default=None,
                        help='Process only a single episode by index (0-based). Useful for testing.')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging for tracking videos')
    parser.add_argument('--wandb-project', type=str, default='ik-preprocessing',
                        help='W&B project name (default: ik-preprocessing)')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')
    return parser.parse_args()


def init_wandb(args, dataset_size):
    """Initialize Weights & Biases logging if enabled."""
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb flag provided but wandb is not installed. Continuing without wandb.")

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'dataset': 'paszea/ik',
                'total_samples': dataset_size,
                'frame_offset': args.frame_offset,
                'chunk_size': args.chunk_size,
                'gpu_chunk_size': args.gpu_chunk_size,
                'visualize': bool(args.visualize),
                'vis_frames': args.vis_frames,
                'max_frames': args.max_frames if args.max_frames else 'all',
                'initial_dots_cam1': args.dots_cam1,
                'initial_dots_cam2': args.dots_cam2,
            }
        )
        print(f"Weights & Biases initialized: {wandb.run.url}")

    return use_wandb


def save_tracking_data(output_path, image_positions, joint_positions):
    """Save processed tracking data to npz file."""
    print(f"\nSaving processed data...")
    save_start = time.time()
    np.savez(
        output_path,
        image_positions=image_positions,
        joint_positions=joint_positions
    )
    save_time = time.time() - save_start
    print(f"[SAVED] {output_path}")
    print(f"[TIME] Data saving: {save_time:.1f}s")
    return save_time


def create_and_upload_visualizations(args, output_path, video_dir, tracks_cam1, tracks_cam2,
                                      visibility_cam1, visibility_cam2, use_wandb):
    """Create visualization videos and optionally upload to W&B."""
    print("\n" + "=" * 70)
    print("Creating Per-Episode Visualizations")
    print("=" * 70)

    vis_dir = output_path.parent / "tracking_visualizations"
    vis_dir.mkdir(exist_ok=True)

    vis_start = time.time()
    video_files = visualize_tracking_per_episode(
        video_dir=video_dir,
        tracks_cam1=tracks_cam1,
        tracks_cam2=tracks_cam2,
        output_dir=vis_dir,
        max_frames_per_episode=args.vis_frames,
        visibility_cam1=visibility_cam1,
        visibility_cam2=visibility_cam2
    )
    vis_time = time.time() - vis_start

    print(f"\n[SAVED] {len(video_files)} videos -> {vis_dir}/")
    print(f"[TIME] Visualization creation: {vis_time:.1f}s ({vis_time/60:.1f} min)")

    upload_time = 0
    if use_wandb and video_files:
        print("\nUploading episode visualizations to Weights & Biases...")
        upload_start = time.time()
        for video_file in video_files:
            episode_name = video_file.stem
            wandb.log({
                f"tracking_{episode_name}": wandb.Video(str(video_file), fps=30, format="mp4")
            })
            print(f"  Uploaded: {video_file.name}")
        upload_time = time.time() - upload_start
        print("[OK] All videos uploaded to W&B")
        print(f"[TIME] Video upload: {upload_time:.1f}s")

    return vis_time, upload_time


def print_timing_summary(timings):
    """Print timing summary for all operations."""
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"Dataset loading:  {timings['dataset']:7.1f}s ({timings['dataset']/60:5.1f} min)")
    print(f"Processing:       {timings['processing']:7.1f}s ({timings['processing']/60:5.1f} min)")
    print(f"Data saving:      {timings['save']:7.1f}s")
    if 'visualization' in timings:
        print(f"Visualization:    {timings['visualization']:7.1f}s ({timings['visualization']/60:5.1f} min)")
        if 'upload' in timings and timings['upload'] > 0:
            print(f"W&B upload:       {timings['upload']:7.1f}s")
    print(f"{'-'*70}")
    print(f"TOTAL TIME:       {timings['total']:7.1f}s ({timings['total']/60:5.1f} min)")
    print("=" * 70)


def finalize_wandb(use_wandb, output_path, image_positions, joint_positions, timings):
    """Log final summary to W&B and finish the run."""
    if not use_wandb:
        return

    wandb.summary['total_frames'] = len(image_positions)
    wandb.summary['image_positions_shape'] = str(image_positions.shape)
    wandb.summary['joint_positions_shape'] = str(joint_positions.shape)

    wandb.summary['time_dataset_loading_sec'] = timings['dataset']
    wandb.summary['time_processing_sec'] = timings['processing']
    wandb.summary['time_saving_sec'] = timings['save']
    if 'visualization' in timings:
        wandb.summary['time_visualization_sec'] = timings['visualization']
    if 'upload' in timings:
        wandb.summary['time_upload_sec'] = timings['upload']
    wandb.summary['time_total_sec'] = timings['total']
    wandb.summary['time_total_min'] = timings['total'] / 60

    artifact = wandb.Artifact('processed-data', type='dataset')
    artifact.add_file(str(output_path), name='processed_data.npz')
    wandb.log_artifact(artifact)

    wandb.finish()
    print("\nWeights & Biases logging complete")


def main():
    args = parse_args()
    timings = {}

    script_start_time = time.time()
    print(f"\n[TIME] Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Validate dot positions
    if args.dots_cam1 is None or args.dots_cam2 is None:
        print("\nWARNING: Initial dot positions not provided!")
        print("You need to manually identify the two dots in the first frame of each camera.")
        print("Example: python preprocess.py --dots-cam1 100 150 120 160 --dots-cam2 200 250 220 260")
        return

    initial_dots_cam1 = np.array(args.dots_cam1).reshape(2, 2)
    initial_dots_cam2 = np.array(args.dots_cam2).reshape(2, 2)

    # Setup output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading HuggingFace dataset...")
    dataset_load_start = time.time()
    dataset = load_dataset("paszea/ik", split='train')
    timings['dataset'] = time.time() - dataset_load_start
    print(f"Loaded {len(dataset)} samples")
    print(f"[TIME] Dataset loading: {timings['dataset']:.1f}s")

    # Initialize W&B
    use_wandb = init_wandb(args, len(dataset))

    # Process dataset
    print("\n" + "="*70)
    print("PROCESSING WITH STREAMING (Memory-Efficient Mode)")
    print("="*70)

    processing_start = time.time()
    result = process_dataset_streaming(
        dataset,
        initial_dots_cam1,
        initial_dots_cam2,
        frame_offset=args.frame_offset,
        video_dir=args.video_dir,
        return_visualization_data=True,
        chunk_size=args.chunk_size,
        gpu_chunk_size=args.gpu_chunk_size,
        temp_dir=output_path.parent,
        vis_frames=args.vis_frames,
        max_frames=args.max_frames,
        episode=args.episode
    )
    image_positions, joint_positions, tracks_cam1, tracks_cam2, visibility_cam1, visibility_cam2, video_dir_used = result
    timings['processing'] = time.time() - processing_start
    print(f"\n[TIME] Total processing (tracking + data prep): {timings['processing']:.1f}s ({timings['processing']/60:.1f} min)")

    # Save data
    timings['save'] = save_tracking_data(output_path, image_positions, joint_positions)

    # Create visualizations
    if args.visualize:
        timings['visualization'], timings['upload'] = create_and_upload_visualizations(
            args, output_path, video_dir_used, tracks_cam1, tracks_cam2,
            visibility_cam1, visibility_cam2, use_wandb
        )

    # Summary
    timings['total'] = time.time() - script_start_time
    print_timing_summary(timings)

    # Finalize W&B
    finalize_wandb(use_wandb, output_path, image_positions, joint_positions, timings)

    print(f"\n[TIME] Script finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
