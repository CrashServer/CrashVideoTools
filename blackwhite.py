#!/usr/bin/env python3
import os
import random
import subprocess
from PIL import Image
import argparse
import tempfile
import shutil

def create_optimized_cache(source_dir, cache_dir, target_size=1080, black_and_white=False):
    """Create a cache directory with optimized images."""

    print(f"\n📦 Creating optimized image cache...")

    # Get all source files
    source_files = get_jpg_files(source_dir)

    if not source_files:
        raise ValueError(f"No JPG files found in {source_dir}")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Get cache suffix based on settings
    cache_suffix = f"_{target_size}{'_bw' if black_and_white else ''}.jpg"

    cached_files = []
    new_files = 0
    existing_files = 0

    print(f"   Cache directory: {cache_dir}")
    print(f"   Target size: {target_size}x{target_size}")
    print(f"   Style: {'Black & White' if black_and_white else 'Color'}")
    print(f"   Processing {len(source_files)} source images...")
    print()

    for i, source_file in enumerate(source_files):
        source_filename = os.path.basename(source_file)
        # Create cache filename based on original name + settings
        cache_filename = os.path.splitext(source_filename)[0] + cache_suffix
        cache_path = os.path.join(cache_dir, cache_filename)

        # Check if cached version already exists and is newer than source
        if (os.path.exists(cache_path) and
            os.path.getmtime(cache_path) > os.path.getmtime(source_file)):
            print(f"   ✓ Cached: {source_filename} → {cache_filename}")
            existing_files += 1
        else:
            # Need to create/update cached version
            original_size = os.path.getsize(source_file)
            print(f"   🔄 Processing: {source_filename}")
            print(f"      Original: {original_size:,} bytes", end="")

            resize_and_optimize_image(source_file, cache_path, target_size, black_and_white)

            optimized_size = os.path.getsize(cache_path)
            compression_ratio = (1 - optimized_size/original_size) * 100
            print(f" → Cached: {optimized_size:,} bytes ({compression_ratio:.1f}% smaller)")
            new_files += 1

        cached_files.append(cache_path)

        if (i + 1) % 10 == 0:
            print(f"\n   Progress: {i + 1}/{len(source_files)} images processed")
            print()

    print(f"\n✅ Cache ready!")
    print(f"   Total cached files: {len(cached_files)}")
    print(f"   New files created: {new_files}")
    print(f"   Existing files reused: {existing_files}")

    return cached_files
    """Resize and optimize image to target size with good compression."""
    # Open and convert to RGB
    img = Image.open(source_path).convert('RGB')

    # Convert to black and white if requested
    if black_and_white:
        img = img.convert('L').convert('RGB')  # Convert to grayscale then back to RGB

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

def resize_and_optimize_image(source_path, output_path, target_size=1080, black_and_white=False):
    """Resize and optimize image to target size with good compression."""
    # Open and convert to RGB
    img = Image.open(source_path).convert('RGB')

    # Convert to black and white if requested
    if black_and_white:
        img = img.convert('L').convert('RGB')  # Convert to grayscale then back to RGB

    # Resize to target size (square)
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Save with optimization
    img.save(output_path, 'JPEG', quality=85, optimize=True)

def create_random_sequence_with_timing(source_dir, num_frames=1080, target_size=1080, black_and_white=False, freeze_probability=0.15):
    """Create a random sequence with varied timing and freeze frames (fallback method)."""

    print(f"\n🎲 Creating random sequence with dynamic timing...")
    if black_and_white:
        print(f"   🎨 Black & white mode enabled")

    # Get all image files
    source_files = get_jpg_files(source_dir)

    if not source_files:
        raise ValueError(f"No JPG files found in {source_dir}")

    print(f"\n📁 Creating temporary directory for optimized frames...")
    temp_dir = tempfile.mkdtemp()
    print(f"   Temp directory: {temp_dir}")

    try:
        print(f"\n🎬 Generating {num_frames} frames with dynamic timing...")
        print(f"   Source pool: {len(source_files)} images")
        print(f"   Freeze frame probability: {freeze_probability*100:.1f}%")
        print(f"   Image processing: {'B&W' if black_and_white else 'Color'} at {target_size}x{target_size}")
        print()

        current_image = None
        current_image_frames = 0
        freeze_duration = 0

        for i in range(num_frames):
            frame_filename = f"frame_{i:06d}.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)

            # Decide if we should continue with current image or pick a new one
            should_change_image = (
                current_image is None or  # First frame
                current_image_frames <= 0 or  # Current image duration expired
                (freeze_duration <= 0 and random.random() < 0.3)  # Random change (30% chance)
            )

            if should_change_image:
                # Pick new image
                current_image = random.choice(source_files)

                # Determine how long to show this image
                if random.random() < freeze_probability:
                    # Freeze frame: show for 3-10 frames
                    freeze_duration = random.randint(3, 10)
                    current_image_frames = freeze_duration
                    frame_type = "FREEZE"
                else:
                    # Normal frame: show for 1-3 frames
                    current_image_frames = random.randint(1, 3)
                    freeze_duration = 0
                    frame_type = "NORMAL"
            else:
                frame_type = "FREEZE" if freeze_duration > 0 else "CONTINUE"

            source_filename = os.path.basename(current_image)

            # Get original file size (only for new images)
            if should_change_image:
                original_size = os.path.getsize(current_image)
                print(f"   Frame {i+1:3d}/{num_frames}: {frame_filename} ← {source_filename} [{frame_type}]")
                print(f"                    Original: {original_size:,} bytes", end="")
            else:
                print(f"   Frame {i+1:3d}/{num_frames}: {frame_filename} ← {source_filename} [{frame_type}]")
                print(f"                    Reusing optimized image", end="")

            # Resize and optimize (or copy if freeze frame)
            if should_change_image:
                resize_and_optimize_image(current_image, frame_path, target_size, black_and_white)
                optimized_size = os.path.getsize(frame_path)
                compression_ratio = (1 - optimized_size/original_size) * 100
                print(f" → Optimized: {optimized_size:,} bytes ({compression_ratio:.1f}% smaller)")
            else:
                # Copy the last frame for freeze effect
                last_frame_path = os.path.join(temp_dir, f"frame_{i-1:06d}.jpg")
                if os.path.exists(last_frame_path):
                    shutil.copy2(last_frame_path, frame_path)
                    copied_size = os.path.getsize(frame_path)
                    print(f" → Copied: {copied_size:,} bytes")

            # Update counters
            current_image_frames -= 1
            if freeze_duration > 0:
                freeze_duration -= 1

            # Progress update
            if (i + 1) % 20 == 0:
                print(f"\n✓ Progress: {i + 1}/{num_frames} frames completed")
                remaining = num_frames - (i + 1)
                if remaining > 0:
                    print(f"   Remaining: {remaining} frames")
                print()

        print(f"\n✅ Dynamic sequence complete!")
        print(f"   Total frames created: {num_frames}")
        print(f"   Style: {'Black & White' if black_and_white else 'Color'}")
        print(f"   Frame size: {target_size}x{target_size}")
        print(f"   Stored in: {temp_dir}")

        return temp_dir

    except Exception as e:
        print(f"\n❌ Error during frame creation: {e}")
        print(f"   Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)
        raise e

def create_random_sequence_from_cache(cached_files, num_frames=1080, freeze_probability=0.15):
    """Create a random sequence with varied timing from cached optimized files."""

    print(f"\n🎲 Creating random sequence from cached files...")

    print(f"\n📁 Creating temporary directory for sequence...")
    temp_dir = tempfile.mkdtemp()
    print(f"   Temp directory: {temp_dir}")

    try:
        print(f"\n🎬 Generating {num_frames} frames with dynamic timing...")
        print(f"   Cached pool: {len(cached_files)} optimized images")
        print(f"   Freeze frame probability: {freeze_probability*100:.1f}%")
        print()

        current_image = None
        current_image_frames = 0
        freeze_duration = 0

        for i in range(num_frames):
            frame_filename = f"frame_{i:06d}.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)

            # Decide if we should continue with current image or pick a new one
            should_change_image = (
                current_image is None or  # First frame
                current_image_frames <= 0 or  # Current image duration expired
                (freeze_duration <= 0 and random.random() < 0.3)  # Random change (30% chance)
            )

            if should_change_image:
                # Pick new image from cache
                current_image = random.choice(cached_files)

                # Determine how long to show this image
                if random.random() < freeze_probability:
                    # Freeze frame: show for 3-10 frames
                    freeze_duration = random.randint(3, 10)
                    current_image_frames = freeze_duration
                    frame_type = "FREEZE"
                else:
                    # Normal frame: show for 1-3 frames
                    current_image_frames = random.randint(1, 3)
                    freeze_duration = 0
                    frame_type = "NORMAL"
            else:
                frame_type = "FREEZE" if freeze_duration > 0 else "CONTINUE"

            source_filename = os.path.basename(current_image)

            # Copy from cache (super fast!)
            print(f"   Frame {i+1:3d}/{num_frames}: {frame_filename} ← {source_filename} [{frame_type}]")
            shutil.copy2(current_image, frame_path)

            copied_size = os.path.getsize(frame_path)
            print(f"                    Copied from cache: {copied_size:,} bytes")

            # Update counters
            current_image_frames -= 1
            if freeze_duration > 0:
                freeze_duration -= 1

            # Progress update
            if (i + 1) % 20 == 0:
                print(f"\n✓ Progress: {i + 1}/{num_frames} frames completed")
                remaining = num_frames - (i + 1)
                if remaining > 0:
                    print(f"   Remaining: {remaining} frames")
                print()

        print(f"\n✅ Sequence from cache complete!")
        print(f"   Total frames created: {num_frames}")
        print(f"   All frames copied from cache (super fast!)")
        print(f"   Stored in: {temp_dir}")

        return temp_dir

    except Exception as e:
        print(f"\n❌ Error during sequence creation: {e}")
        print(f"   Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)
        raise e
def create_random_sequence_from_cache(cached_files, num_frames=1080, freeze_probability=0.15):
    """Create a random sequence with varied timing from cached optimized files."""

    print(f"\n🎲 Creating random sequence from cached files...")

    print(f"\n📁 Creating temporary directory for sequence...")
    temp_dir = tempfile.mkdtemp()
    print(f"   Temp directory: {temp_dir}")

    try:
        print(f"\n🎬 Generating {num_frames} frames with dynamic timing...")
        print(f"   Cached pool: {len(cached_files)} optimized images")
        print(f"   Freeze frame probability: {freeze_probability*100:.1f}%")
        print()

        current_image = None
        current_image_frames = 0
        freeze_duration = 0

        for i in range(num_frames):
            frame_filename = f"frame_{i:06d}.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)

            # Decide if we should continue with current image or pick a new one
            should_change_image = (
                current_image is None or  # First frame
                current_image_frames <= 0 or  # Current image duration expired
                (freeze_duration <= 0 and random.random() < 0.3)  # Random change (30% chance)
            )

            if should_change_image:
                # Pick new image from cache
                current_image = random.choice(cached_files)

                # Determine how long to show this image
                if random.random() < freeze_probability:
                    # Freeze frame: show for 3-10 frames
                    freeze_duration = random.randint(3, 10)
                    current_image_frames = freeze_duration
                    frame_type = "FREEZE"
                else:
                    # Normal frame: show for 1-3 frames
                    current_image_frames = random.randint(1, 3)
                    freeze_duration = 0
                    frame_type = "NORMAL"
            else:
                frame_type = "FREEZE" if freeze_duration > 0 else "CONTINUE"

            source_filename = os.path.basename(current_image)

            # Copy from cache (super fast!)
            print(f"   Frame {i+1:3d}/{num_frames}: {frame_filename} ← {source_filename} [{frame_type}]")
            shutil.copy2(current_image, frame_path)

            copied_size = os.path.getsize(frame_path)
            print(f"                    Copied from cache: {copied_size:,} bytes")

            # Update counters
            current_image_frames -= 1
            if freeze_duration > 0:
                freeze_duration -= 1

            # Progress update
            if (i + 1) % 20 == 0:
                print(f"\n✓ Progress: {i + 1}/{num_frames} frames completed")
                remaining = num_frames - (i + 1)
                if remaining > 0:
                    print(f"   Remaining: {remaining} frames")
                print()

        print(f"\n✅ Sequence from cache complete!")
        print(f"   Total frames created: {num_frames}")
        print(f"   All frames copied from cache (super fast!)")
        print(f"   Stored in: {temp_dir}")

        return temp_dir

    except Exception as e:
        print(f"\n❌ Error during sequence creation: {e}")
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
    parser.add_argument('--bw', '--black-white', action='store_true', help='Convert to black and white')
    parser.add_argument('--freeze-prob', type=float, default=0.15, help='Probability of freeze frames (0.0-1.0, default: 0.15)')
    parser.add_argument('--cache-dir', default='optimized_cache', help='Directory to store optimized images (default: optimized_cache)')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache and process images directly (slower)')

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
    print(f"   Style: {'Black & White' if args.bw else 'Color'}")
    print(f"   Freeze probability: {args.freeze_prob*100:.1f}%")
    print(f"   Cache: {'Disabled' if args.no_cache else args.cache_dir}")

    try:
        if args.no_cache:
            # Old method: process images directly (slower but no cache)
            print(f"\n⚠️  Cache disabled - processing images directly (slower)")
            temp_dir = create_random_sequence_with_timing(
                args.input_dir,
                args.frames,
                target_size=1080,
                black_and_white=args.bw,
                freeze_probability=args.freeze_prob
            )
        else:
            # New method: use cache for speed
            # Create or update cache
            cached_files = create_optimized_cache(
                args.input_dir,
                args.cache_dir,
                target_size=1080,
                black_and_white=args.bw
            )

            # Create sequence from cache (super fast!)
            temp_dir = create_random_sequence_from_cache(
                cached_files,
                args.frames,
                freeze_probability=args.freeze_prob
            )

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
