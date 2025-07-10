#!/usr/bin/env python3
import os
import random
import subprocess
from PIL import Image
import argparse
import tempfile
import shutil

def resize_and_optimize_image(source_path, output_path, target_size=1080):
    """Resize and optimize image to target size with good compression."""
    # Open and convert to RGB
    img = Image.open(source_path).convert('RGB')

    # Resize to target size (square)
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Save with optimization
    img.save(output_path, 'JPEG', quality=85, optimize=True)

def get_jpg_files(directory):
    """Get all JPG files from a directory."""
    print(f"\n🔍 Scanning directory: {directory}")
    jpg_files = []
    all_files = os.listdir(directory)
    print(f"   Found {len(all_files)} total files")

    for file in all_files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            full_path = os.path.join(directory, file)
            jpg_files.append(full_path)
            print(f"   ✓ Added: {file}")
        else:
            print(f"   ✗ Skipped: {file} (not JPG)")

    print(f"📊 Total JPG files found: {len(jpg_files)}")
    return jpg_files

def create_random_sequence(source_dir, num_frames=1080, target_size=1080):
    """Create a random sequence of optimized frames from source directory."""

    print(f"\n🎲 Creating random sequence with image optimization...")

    # Get all image files
    source_files = get_jpg_files(source_dir)

    if not source_files:
        raise ValueError(f"No JPG files found in {source_dir}")

    print(f"\n📁 Creating temporary directory for optimized frames...")
    # Create temporary directory for sequenced frames
    temp_dir = tempfile.mkdtemp()
    print(f"   Temp directory: {temp_dir}")

    try:
        print(f"\n🎬 Generating {num_frames} random frames at {target_size}x{target_size}...")
        print(f"   Source pool: {len(source_files)} images")
        print(f"   Each frame will be resized and optimized")
        print()

        for i in range(num_frames):
            # Randomly select source image
            source_file = random.choice(source_files)
            source_filename = os.path.basename(source_file)

            # Create frame filename
            frame_filename = f"frame_{i:06d}.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)

            # Get original file size
            original_size = os.path.getsize(source_file)

            # Resize and optimize the image
            print(f"   Frame {i+1:3d}/{num_frames}: {frame_filename} ← {source_filename}")
            print(f"                    Original: {original_size:,} bytes", end="")

            resize_and_optimize_image(source_file, frame_path, target_size)

            # Show optimized file size
            optimized_size = os.path.getsize(frame_path)
            compression_ratio = (1 - optimized_size/original_size) * 100
            print(f" → Optimized: {optimized_size:,} bytes ({compression_ratio:.1f}% smaller)")

            if (i + 1) % 10 == 0:
                print(f"\n✓ Progress: {i + 1}/{num_frames} frames completed")
                remaining = num_frames - (i + 1)
                if remaining > 0:
                    print(f"   Remaining: {remaining} frames")
                print()

        print(f"\n✅ Random sequence with optimization complete!")
        print(f"   Total frames created: {num_frames}")
        print(f"   All frames resized to: {target_size}x{target_size}")
        print(f"   Stored in: {temp_dir}")

        return temp_dir

    except Exception as e:
        print(f"\n❌ Error during frame creation: {e}")
        print(f"   Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)
        raise e

def create_video_with_ffmpeg(frame_dir, output_path, fps=24):
    """Create 1080x1080 video from frames using ffmpeg with Instagram-compatible encoding."""

    print(f"\n🎥 Creating 1080x1080 Instagram-compatible video...")

    # Count frames in directory
    frame_files = [f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.jpg')]
    frame_count = len(frame_files)
    duration = frame_count / fps

    print(f"   Input frames: {frame_count}")
    print(f"   Frame rate: {fps} fps")
    print(f"   Video duration: {duration:.2f} seconds")
    print(f"   Output resolution: 1080x1080 (Instagram square)")
    print(f"   Output file: {output_path}")

    # Instagram-optimized encoding settings for 1080x1080
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-framerate', str(fps),
        '-i', os.path.join(frame_dir, 'frame_%06d.jpg'),

        # Force 1080x1080 output (in case input isn't exactly that)
        '-vf', 'scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2',

        # Video codec settings for Instagram compatibility
        '-c:v', 'libx264',
        '-profile:v', 'baseline',  # Instagram prefers baseline profile
        '-level', '3.1',           # Level 3.1 for 1080p
        '-pix_fmt', 'yuv420p',     # Required for Instagram

        # Quality and bitrate settings optimized for 1080p
        '-crf', '21',              # Good quality for 1080p
        '-maxrate', '8M',          # Instagram-friendly bitrate for 1080p
        '-bufsize', '16M',         # Buffer size

        # Frame settings
        '-r', str(fps),            # Output framerate
        '-g', str(fps * 2),        # Keyframe interval (2 seconds)

        # Audio (silent video)
        '-an',                     # No audio

        # Optimization flags
        '-movflags', '+faststart', # Web optimization
        '-preset', 'medium',       # Encoding speed vs compression

        output_path
    ]

    print(f"\n🔧 Instagram-optimized 1080p FFmpeg settings:")
    print(f"   Resolution: 1080x1080 (forced)")
    print(f"   Profile: baseline (Instagram compatible)")
    print(f"   Level: 3.1 (1080p compatible)")
    print(f"   Pixel format: yuv420p (required)")
    print(f"   Max bitrate: 8Mbps (Instagram optimized)")
    print(f"   CRF: 21 (good quality for 1080p)")
    print(f"   Preset: medium (good compression)")
    print(f"   Faststart: enabled (web streaming)")
    print(f"\n   Full command:")
    print(f"   {' '.join(cmd)}")
    print(f"\n⚙️  Running FFmpeg...")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Show relevant ffmpeg output
        if result.stderr:
            print(f"\n📝 FFmpeg encoding info:")
            for line in result.stderr.split('\n'):
                if any(keyword in line.lower() for keyword in ['frame=', 'fps=', 'bitrate=', 'size=']):
                    print(f"   {line.strip()}")

        # Check output file and validate
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            print(f"\n✅ 1080x1080 video created successfully!")
            print(f"   File: {output_path}")
            print(f"   Size: {output_size:,} bytes ({output_size/1024/1024:.1f} MB)")

            # Check if file size is reasonable for Instagram
            max_size_mb = 100  # Instagram limit is 100MB for videos
            actual_size_mb = output_size/1024/1024
            if actual_size_mb > max_size_mb:
                print(f"   ⚠️  Warning: File size ({actual_size_mb:.1f}MB) exceeds Instagram limit ({max_size_mb}MB)")
                print(f"       Consider reducing frames, duration, or increasing CRF value")
            else:
                print(f"   ✓ File size OK for Instagram (under {max_size_mb}MB limit)")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ FFmpeg failed!")
        print(f"   Error code: {e.returncode}")
        if e.stderr:
            print(f"   Error output:")
            for line in e.stderr.split('\n'):
                if line.strip():
                    print(f"     {line}")

        return False

def main():
    parser = argparse.ArgumentParser(description='Create video with randomly ordered frames from a directory')
    parser.add_argument('input_dir', help='Directory containing JPG images')
    parser.add_argument('-o', '--output', default='random_animation.mp4', help='Output video filename')
    parser.add_argument('-f', '--frames', type=int, default=1080, help='Number of frames in output video (default: 1080 for 45s at 24fps)')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second')
    parser.add_argument('--duration', type=float, default=45.0, help='Video duration in seconds (default: 45)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')

    args = parser.parse_args()

    print("🎬 Random Frame Animation Generator")
    print("=" * 40)

    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")
    else:
        print(f"🎲 Using random seed: None (truly random)")

    # Validate directory
    print(f"\n📂 Input validation...")
    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: Directory '{args.input_dir}' does not exist")
        return 1

    print(f"✓ Input directory exists: {args.input_dir}")

    # Calculate frames based on duration (always use duration now)
    args.frames = int(args.duration * args.fps)
    print(f"\n⏱️  Video settings:")
    print(f"   Duration: {args.duration}s")
    print(f"   Frame rate: {args.fps}fps")
    print(f"   Total frames: {args.frames}")

    try:
        # Create random sequence with optimization
        temp_dir = create_random_sequence(args.input_dir, args.frames, target_size=1080)

        # Create video
        success = create_video_with_ffmpeg(temp_dir, args.output, args.fps)

        # Clean up
        print(f"\n🧹 Cleaning up...")
        print(f"   Removing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print(f"   ✓ Temporary files cleaned up")

        if success:
            print(f"\n🎉 Animation complete!")
            print(f"   Output file: {args.output}")
            print(f"   You can now play your video!")
            return 0
        else:
            print(f"\n❌ Failed to create video")
            return 1

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"   Make sure:")
        print(f"   - FFmpeg is installed and in your PATH")
        print(f"   - Input directory contains JPG files")
        print(f"   - You have write permissions for output directory")
        return 1

if __name__ == "__main__":
    exit(main())
