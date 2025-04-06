import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class VideoLoopGenerator:
    def __init__(self, chunk_size=300, overlap_frames=15, similarity_threshold=10.0, use_gpu=True):
        """
        Initialize the video loop generator.
        
        Args:
            chunk_size: Number of frames to process at once
            overlap_frames: Number of frames to blend for smooth transitions
            similarity_threshold: Threshold for determining similar frames (lower = more similar)
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.similarity_threshold = similarity_threshold
        
        # Check if GPU is available and requested
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        if self.use_gpu:
            print(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU processing")
            if use_gpu and not torch.cuda.is_available():
                print("GPU requested but not available, falling back to CPU")
        
        # Initialize optical flow calculator - we'll still use this for CPU processing
        self.flow_calculator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        
        # Initialize the frame feature extractor model if using GPU
        if self.use_gpu:
            self.initialize_feature_extractor()
    
    def load_video_info(self, video_path):
        """Get video information without loading all frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"Video info: {self.width}x{self.height}, {self.fps} fps, {self.total_frames} frames")
        return self.fps, self.total_frames, self.width, self.height
    
    def get_frame_chunk(self, video_path, start_frame, end_frame):
        """Load a chunk of frames from the video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(min(end_frame - start_frame, self.total_frames - start_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def initialize_feature_extractor(self):
        """Initialize a simple CNN for feature extraction using PyTorch."""
        # We'll use a simplified ResNet-like architecture
        class FeatureExtractor(nn.Module):
            def __init__(self):
                super(FeatureExtractor, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(128)
                
                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn3 = nn.BatchNorm2d(256)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                
                x = self.conv3(x)
                x = self.bn3(x)
                x = self.relu(x)
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x
        
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.feature_extractor.eval()
    
    def compute_frame_difference(self, frame1, frame2):
        """Compute the visual difference between two frames."""
        if self.use_gpu:
            return self.compute_frame_difference_gpu(frame1, frame2)
        else:
            return self.compute_frame_difference_cpu(frame1, frame2)
    
    def compute_frame_difference_cpu(self, frame1, frame2):
        """Compute the visual difference between two frames using optical flow on CPU."""
        # Convert to grayscale for optical flow
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate flow
        flow = self.flow_calculator.calc(gray1, gray2, None)
        
        # Calculate flow magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)
    
    def compute_frame_difference_gpu(self, frame1, frame2):
        """Compute the visual difference between two frames using feature extraction on GPU."""
        # Convert frames to PyTorch tensors
        tensor1 = self.frame_to_tensor(frame1).to(self.device)
        tensor2 = self.frame_to_tensor(frame2).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features1 = self.feature_extractor(tensor1)
            features2 = self.feature_extractor(tensor2)
        
        # Compute cosine similarity (higher means more similar)
        similarity = F.cosine_similarity(features1, features2)
        # Convert to a difference measure (lower means more similar)
        difference = 1.0 - similarity.item()
        
        return difference * 10.0  # Scale to be roughly in the same range as the optical flow method
    
    def frame_to_tensor(self, frame):
        """Convert a frame to a PyTorch tensor suitable for the model."""
        # Resize to a standard size to speed up processing
        resized = cv2.resize(frame, (224, 224))
        # Convert from BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        # Convert to tensor with batch dimension [1, 3, H, W]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def find_loop_candidates(self, frames, min_loop_length=30):
        """Find potential loop points in a sequence of frames."""
        candidates = []
        total_comparisons = (len(frames) - min_loop_length) * len(frames) // 20  # Approximate
        
        # If using GPU, we can batch process comparisons for faster results
        if self.use_gpu and len(frames) > 50:
            return self.find_loop_candidates_gpu_batched(frames, min_loop_length)
        
        # We'll compare frames with a sliding window approach
        # Start comparing from min_loop_length to give enough content for a loop
        with tqdm(total=total_comparisons, desc="Finding loop candidates") as pbar:
            for start_idx in range(len(frames) - min_loop_length):
                start_frame = frames[start_idx]
                
                # Compare with frames further ahead to find potential loop points
                for end_idx in range(start_idx + min_loop_length, len(frames), max(1, (len(frames) - start_idx) // 20)):
                    end_frame = frames[end_idx]
                    
                    # Calculate visual difference
                    diff = self.compute_frame_difference(start_frame, end_frame)
                    
                    # If frames are similar enough, add as a candidate
                    if diff < self.similarity_threshold:
                        loop_length = end_idx - start_idx
                        candidates.append((start_idx, end_idx, diff, loop_length))
                    
                    pbar.update(1)
        
        # Sort by difference (lower is better) and then by loop length (longer is better)
        return sorted(candidates, key=lambda x: (x[2], -x[3]))
    
    def find_loop_candidates_gpu_batched(self, frames, min_loop_length=30):
        """Find loop candidates using GPU-accelerated batch processing."""
        candidates = []
        start_time = time.time()
        
        # Create feature vectors for all frames in batches
        print("Generating frame features in batches...")
        frame_features = []
        batch_size = 32  # Process 32 frames at a time
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_tensors = torch.cat([self.frame_to_tensor(frame) for frame in batch_frames]).to(self.device)
            
            with torch.no_grad():
                batch_features = self.feature_extractor(batch_tensors)
            
            frame_features.append(batch_features)
        
        # Concatenate all features
        all_features = torch.cat(frame_features)
        print(f"Generated features for {len(frames)} frames in {time.time() - start_time:.2f} seconds")
        
        # Now compute similarities between frames
        print("Computing frame similarities...")
        start_time = time.time()
        
        # For each start frame, compare with potential end frames
        with tqdm(total=len(frames) - min_loop_length, desc="Processing frames") as pbar:
            for start_idx in range(len(frames) - min_loop_length):
                start_feature = all_features[start_idx].unsqueeze(0)
                
                # Select potential end frames (ones that are at least min_loop_length away)
                end_indices = list(range(start_idx + min_loop_length, len(frames), max(1, (len(frames) - start_idx - min_loop_length) // 20)))
                if not end_indices:
                    continue
                
                end_features = all_features[end_indices]
                
                # Compute similarities in one batch operation
                similarities = F.cosine_similarity(start_feature, end_features)
                differences = 1.0 - similarities.cpu().numpy()
                differences *= 10.0  # Scale to be in the same range as optical flow method
                
                # Add candidates that meet the threshold
                for i, end_idx in enumerate(end_indices):
                    diff = differences[i]
                    if diff < self.similarity_threshold:
                        loop_length = end_idx - start_idx
                        candidates.append((start_idx, end_idx, diff, loop_length))
                
                pbar.update(1)
        
        print(f"Found {len(candidates)} candidates in {time.time() - start_time:.2f} seconds")
        
        # Sort by difference (lower is better) and then by loop length (longer is better)
        return sorted(candidates, key=lambda x: (x[2], -x[3]))
    
    def create_smooth_loop(self, frames, start_idx, end_idx):
        """Create a smooth loop by blending the transition frames."""
        loop_frames = frames[start_idx:end_idx].copy()
        blend_frames = min(self.overlap_frames, len(loop_frames) // 4)  # Don't use too many blend frames for short loops
        
        if self.use_gpu and blend_frames > 0:
            return self.create_smooth_loop_gpu(frames, start_idx, end_idx, blend_frames)
        
        # CPU-based blending
        for i in range(blend_frames):
            alpha = i / blend_frames
            last_idx = len(loop_frames) - blend_frames + i
            
            # Create a smooth transition between the end and start
            blended = cv2.addWeighted(
                loop_frames[last_idx], 
                1 - alpha,
                loop_frames[i], 
                alpha, 
                0
            )
            loop_frames[last_idx] = blended
        
        return loop_frames
    
    def create_smooth_loop_gpu(self, frames, start_idx, end_idx, blend_frames):
        """Create a smooth loop by blending the transition frames using GPU."""
        loop_frames = frames[start_idx:end_idx].copy()
        
        # Convert relevant frames to tensors
        start_tensors = []
        end_tensors = []
        
        for i in range(blend_frames):
            # Get the frames to blend
            start_frame = frames[start_idx + i]
            end_frame = frames[end_idx - blend_frames + i]
            
            # Convert to tensors [C, H, W]
            start_tensor = torch.from_numpy(start_frame).float().permute(2, 0, 1) / 255.0
            end_tensor = torch.from_numpy(end_frame).float().permute(2, 0, 1) / 255.0
            
            start_tensors.append(start_tensor)
            end_tensors.append(end_tensor)
        
        # Stack tensors into batches [B, C, H, W]
        start_batch = torch.stack(start_tensors).to(self.device)
        end_batch = torch.stack(end_tensors).to(self.device)
        
        # Create weights for blending
        alphas = torch.linspace(0, 1, blend_frames).view(-1, 1, 1, 1).to(self.device)
        
        # Blend frames in one operation
        blended_batch = (1 - alphas) * end_batch + alphas * start_batch
        
        # Convert back to numpy and update frames
        blended_frames = (blended_batch.cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
        
        # Update the loop frames with blended transitions
        for i in range(blend_frames):
            last_idx = len(loop_frames) - blend_frames + i
            loop_frames[last_idx] = blended_frames[i]
        
        return loop_frames
    
    def save_loop_video(self, frames, output_path, fps=None):
        """Save the loop frames as a video."""
        if fps is None:
            fps = self.fps
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Saved loop video to {output_path}")
    
    def process_video(self, video_path, output_path, min_loop_length=30, max_candidates=5):
        """Process the whole video to find and create loops."""
        self.load_video_info(video_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Process video in chunks to avoid memory issues
        for chunk_start in range(0, self.total_frames, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.total_frames)
            print(f"Processing chunk {chunk_start}-{chunk_end} of {self.total_frames}")
            
            # Get frames for this chunk
            frames = self.get_frame_chunk(video_path, chunk_start, chunk_end)
            if not frames:
                continue
                
            # Find loop candidates in this chunk
            candidates = self.find_loop_candidates(frames, min_loop_length)
            
            # Process the top candidates
            for idx, (start_idx, end_idx, diff, length) in enumerate(candidates[:max_candidates]):
                print(f"Candidate {idx+1}: frames {start_idx}-{end_idx}, difference: {diff:.2f}, length: {length}")
                
                # Create the loop
                loop_frames = self.create_smooth_loop(frames, start_idx, end_idx)
                
                # Save the loop
                candidate_output = output_path.replace('.mp4', f'_chunk{chunk_start}_candidate{idx+1}.mp4')
                self.save_loop_video(loop_frames, candidate_output)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Generate looping videos from a source video')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('--output', '-o', default='output_loop.mp4', help='Output video file path')
    parser.add_argument('--chunk-size', '-c', type=int, default=300, help='Number of frames to process at once')
    parser.add_argument('--min-loop-length', '-m', type=int, default=30, help='Minimum loop length in frames')
    parser.add_argument('--threshold', '-t', type=float, default=10.0, help='Similarity threshold (lower = more similar)')
    parser.add_argument('--blend-frames', '-b', type=int, default=15, help='Number of frames to blend for transitions')
    parser.add_argument('--candidates', '-n', type=int, default=3, help='Maximum number of loop candidates to generate per chunk')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing even if GPU is available')
    parser.add_argument('--benchmark', action='store_true', help='Run CPU vs GPU benchmark on this video')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(args)
        return
    
    generator = VideoLoopGenerator(
        chunk_size=args.chunk_size,
        overlap_frames=args.blend_frames,
        similarity_threshold=args.threshold,
        use_gpu=not args.cpu
    )
    
    generator.process_video(
        args.input,
        args.output,
        min_loop_length=args.min_loop_length,
        max_candidates=args.candidates
    )


def run_benchmark(args):
    """Run a CPU vs GPU benchmark on the same video."""
    print("=" * 50)
    print("Running CPU vs GPU benchmark")
    print("=" * 50)
    
    video_path = args.input
    test_frames = 300  # Use the first 300 frames for benchmarking
    
    # Create a CPU instance
    print("\nInitializing CPU processor...")
    cpu_generator = VideoLoopGenerator(
        chunk_size=test_frames,
        overlap_frames=args.blend_frames,
        similarity_threshold=args.threshold,
        use_gpu=False
    )
    
    # Load the test frames
    print(f"\nLoading {test_frames} frames from video for benchmark...")
    cpu_generator.load_video_info(video_path)
    frames = cpu_generator.get_frame_chunk(video_path, 0, test_frames)
    print(f"Loaded {len(frames)} frames")
    
    # Benchmark CPU
    print("\nBenchmarking CPU processing...")
    cpu_start = time.time()
    cpu_generator.find_loop_candidates(frames, min_loop_length=args.min_loop_length)
    cpu_time = time.time() - cpu_start
    print(f"CPU processing time: {cpu_time:.2f} seconds")
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("\nGPU not available for benchmarking")
        return
    
    # Create a GPU instance
    print("\nInitializing GPU processor...")
    gpu_generator = VideoLoopGenerator(
        chunk_size=test_frames,
        overlap_frames=args.blend_frames,
        similarity_threshold=args.threshold,
        use_gpu=True
    )
    
    # Benchmark GPU
    print("\nBenchmarking GPU processing...")
    gpu_start = time.time()
    gpu_generator.find_loop_candidates(frames, min_loop_length=args.min_loop_length)
    gpu_time = time.time() - gpu_start
    print(f"GPU processing time: {gpu_time:.2f} seconds")
    
    # Print results
    print("\n" + "=" * 50)
    print("Benchmark Results:")
    print(f"CPU time: {cpu_time:.2f} seconds")
    print(f"GPU time: {gpu_time:.2f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()