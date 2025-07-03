"""
Script Generator
Generates bash scripts for video montage creation
"""

import os

class ScriptGenerator:
    def generate_script(self, settings):
        """Generate bash script with given settings"""
        
        # Extract audio file name
        audio_file = os.path.basename(settings.get('audio_file', '01.mp4'))
        
        # Audio timing parameters
        if settings.get('use_audio_timing', False):
            audio_start = settings.get('audio_start_time', 0)
            audio_end = settings.get('audio_end_time', 0)
            target_duration = audio_end - audio_start
        else:
            audio_start = 0
            audio_end = 0
            target_duration = 0
        
        script = f'''#!/bin/bash
mkdir -p clips

echo "🎬 Creating montage with GUI parameters..."

# Parameters from GUI
AUDIO_FILE="{audio_file}"
MAX_VIDEOS={settings.get('max_videos', 25)}
MAX_IMAGES={settings.get('max_images', 10)}
VIDEO_DURATION={settings.get('video_duration', 1.8)}
IMAGE_DURATION={settings.get('image_duration', 0.3)}
SPEED_MULTIPLIER={settings.get('speed_multiplier', 2.0)}
VIDEO_QUALITY={settings.get('video_quality', 23)}
IMAGE_QUALITY={settings.get('image_quality', 18)}
CONTRAST={settings.get('contrast', 1.2)}
SATURATION={settings.get('saturation', 1.3)}
BRIGHTNESS={settings.get('brightness', 0.0)}
MATCH_AUDIO_LENGTH={"true" if settings.get('match_audio_length', True) else "false"}
RANDOM_SHUFFLE={"true" if settings.get('random_shuffle', True) else "false"}
GLITCH_EFFECTS={"true" if settings.get('glitch_effects', True) else "false"}
IMAGE_FLASH_INTENSITY={settings.get('image_flash_intensity', 3.0)}
OUTPUT_NAME="{settings.get('output_name', 'MONTAGE')}"

# Audio timing parameters
USE_AUDIO_TIMING={"true" if settings.get('use_audio_timing', False) else "false"}
AUDIO_START={audio_start}
AUDIO_END={audio_end}
CUSTOM_DURATION={target_duration}

# Check audio file
if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "❌ Audio file $AUDIO_FILE not found!"
    exit 1
fi

# Determine target duration
if [[ "$USE_AUDIO_TIMING" == "true" ]]; then
    TARGET_DURATION=$(echo "$CUSTOM_DURATION" | cut -d. -f1)
    echo "🎵 Using custom audio timing: ${{AUDIO_START}}s to ${{AUDIO_END}}s (duration: ${{TARGET_DURATION}}s)"
elif [[ "$MATCH_AUDIO_LENGTH" == "true" ]]; then
    FULL_AUDIO_DURATION=$(ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$AUDIO_FILE" 2>/dev/null)
    TARGET_DURATION=$(echo "$FULL_AUDIO_DURATION" | cut -d. -f1)
    echo "🎵 Using full audio duration: ${{TARGET_DURATION}}s"
else
    TARGET_DURATION=50
    echo "🎵 Using fixed duration: ${{TARGET_DURATION}}s"
fi

# Get video files (exclude audio file if it's a video)
video_files=()
for video in *.mp4 *.mov *.mkv *.avi *.MP4 *.MOV *.MKV *.AVI; do
    if [[ -f "$video" && "$video" != "$AUDIO_FILE" ]]; then
        video_files+=("$video")
    fi
done

# Get image files  
image_files=()
for image in *.jpg *.jpeg *.png *.bmp *.tiff *.JPG *.JPEG *.PNG *.BMP *.TIFF; do
    if [[ -f "$image" ]]; then
        image_files+=("$image")
    fi
done

echo "Found: ${{#video_files[@]}} videos, ${{#image_files[@]}} images"

# Process videos
echo "📹 Processing videos..."
video_count=0

for ((i=0; i<${{#video_files[@]}} && i<MAX_VIDEOS; i++)); do
    video="${{video_files[$i]}}"
    echo "Video $((i+1)): $(basename "$video")"
    
    # Build video filter
    video_filter="setpts=PTS/$SPEED_MULTIPLIER,scale=1920:1080,eq=contrast=$CONTRAST:saturation=$SATURATION:brightness=$BRIGHTNESS"
    
    # Add glitch effects randomly
    if [[ "$GLITCH_EFFECTS" == "true" && $((RANDOM % 3)) == 0 ]]; then
        echo "  └─ Adding glitch effect"
        case $((RANDOM % 3)) in
            0) video_filter="$video_filter,hue=h=$((RANDOM % 360)):s=2" ;;
            1) video_filter="$video_filter,eq=contrast=2:saturation=3" ;;
            2) video_filter="$video_filter,noise=alls=5:allf=t" ;;
        esac
    fi
    
    ffmpeg -i "$video" -t $VIDEO_DURATION \\
        -vf "$video_filter" \\
        -af "atempo=$SPEED_MULTIPLIER" \\
        -c:v libx264 -crf $VIDEO_QUALITY -preset fast \\
        -c:a aac -b:a 96k -ar 48000 \\
        "clips/video_$(printf "%03d" $video_count).mp4" -y -loglevel error
    
    if [[ -f "clips/video_$(printf "%03d" $video_count).mp4" ]]; then
        file_size=$(stat -c%s "clips/video_$(printf "%03d" $video_count).mp4" 2>/dev/null || echo "0")
        if [[ $file_size -gt 50000 ]]; then
            echo "  ✅ Created: ${{file_size}} bytes"
            ((video_count++))
        else
            rm -f "clips/video_$(printf "%03d" $video_count).mp4"
        fi
    fi
done

# Process images
echo "🖼️  Processing images..."
image_count=0

for ((i=0; i<${{#image_files[@]}} && i<MAX_IMAGES; i++)); do
    image="${{image_files[$i]}}"
    
    # Build image filter based on flash intensity
    base_contrast=$(echo "$IMAGE_FLASH_INTENSITY * 0.8" | bc -l)
    base_brightness=$(echo "$IMAGE_FLASH_INTENSITY * 0.1" | bc -l)
    base_saturation=$(echo "$IMAGE_FLASH_INTENSITY * 0.6" | bc -l)
    
    effect=$((RANDOM % 4))
    case $effect in
        0) image_filter="scale=1920:1080,eq=contrast=$base_contrast:brightness=$base_brightness:saturation=$base_saturation" ;;
        1) image_filter="scale=1920:1080,hue=h=$((RANDOM % 360)):s=$base_saturation,eq=contrast=$base_contrast" ;;
        2) image_filter="scale=1920:1080,eq=contrast=$base_contrast:saturation=$base_saturation,negate" ;;
        3) image_filter="scale=1920:1080,eq=contrast=$base_contrast:brightness=$base_brightness:saturation=$base_saturation" ;;
    esac
    
    ffmpeg -loop 1 -i "$image" -t $IMAGE_DURATION \\
        -vf "$image_filter" \\
        -c:v libx264 -crf $IMAGE_QUALITY -preset fast \\
        -r 30 -an \\
        "clips/image_$(printf "%03d" $image_count).mp4" -y -loglevel error
    
    if [[ -f "clips/image_$(printf "%03d" $image_count).mp4" ]]; then
        file_size=$(stat -c%s "clips/image_$(printf "%03d" $image_count).mp4" 2>/dev/null || echo "0")
        if [[ $file_size -gt 10000 ]]; then
            echo "  ✅ Created flash: ${{file_size}} bytes"
            ((image_count++))
        else
            rm -f "clips/image_$(printf "%03d" $image_count).mp4"
        fi
    fi
done

echo "📊 Created: $video_count videos, $image_count images"

if [[ $((video_count + image_count)) -lt 3 ]]; then
    echo "❌ Too few clips created"
    exit 1
fi

# Create sequence
echo "🔗 Creating sequence..."
> sequence.txt

# Create arrays of all clips
all_clips=()
for ((v=0; v<video_count; v++)); do
    all_clips+=("video_$(printf "%03d" $v).mp4")
done
for ((i=0; i<image_count; i++)); do
    all_clips+=("image_$(printf "%03d" $i).mp4")
done

# Shuffle if enabled
if [[ "$RANDOM_SHUFFLE" == "true" ]]; then
    echo "🎲 Shuffling clips randomly..."
    for ((i=0; i<${{#all_clips[@]}}; i++)); do
        j=$((RANDOM % ${{#all_clips[@]}}))
        temp="${{all_clips[i]}}"
        all_clips[i]="${{all_clips[j]}}"
        all_clips[j]="$temp"
    done
else
    echo "📋 Using sequential order..."
fi

# Write sequence
for clip in "${{all_clips[@]}}"; do
    if [[ -f "clips/$clip" ]]; then
        echo "file '$PWD/clips/$clip'" >> sequence.txt
    fi
done

sequence_length=$(wc -l < sequence.txt)
echo "Sequence: $sequence_length clips"

# Extend sequence if needed for target duration
current_estimated_duration=$(echo "$sequence_length * ($VIDEO_DURATION + $IMAGE_DURATION) / 2" | bc -l)
if [[ $(echo "$current_estimated_duration < $TARGET_DURATION" | bc -l) == "1" ]]; then
    echo "⚠️  Extending sequence to reach target duration..."
    loops_needed=$(echo "$TARGET_DURATION / $current_estimated_duration + 1" | bc -l | cut -d. -f1)
    
    cp sequence.txt temp_seq.txt
    for ((loop=1; loop<loops_needed; loop++)); do
        cat temp_seq.txt >> sequence.txt
    done
    rm temp_seq.txt
    echo "Extended to: $(wc -l < sequence.txt) clips"
fi

# Concatenate
echo "🔗 Concatenating clips..."
ffmpeg -f concat -safe 0 -i sequence.txt \\
    -c:v libx264 -crf 20 -preset medium \\
    -c:a aac -b:a 128k -ar 48000 \\
    -r 30 \\
    video_no_music.mp4 -y -loglevel warning

if [[ ! -f video_no_music.mp4 ]]; then
    echo "❌ Video concatenation failed"
    exit 1
fi

# Get final video duration
video_duration=$(ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 video_no_music.mp4 2>/dev/null)
echo "Video duration: ${{video_duration}}s"

# Add audio and finalize with timing
echo "🎵 Adding audio track..."

if [[ "$USE_AUDIO_TIMING" == "true" ]]; then
    echo "  └─ Using custom timing: ${{AUDIO_START}}s to ${{AUDIO_END}}s"
    ffmpeg -i video_no_music.mp4 -ss "$AUDIO_START" -i "$AUDIO_FILE" \\
        -t "$TARGET_DURATION" \\
        -map 0:v -map 1:a \\
        -c:v copy -c:a aac -b:a 192k \\
        -shortest \\
        "${{OUTPUT_NAME}}.mp4" -y -loglevel warning
else
    echo "  └─ Using full audio track"
    ffmpeg -i video_no_music.mp4 -i "$AUDIO_FILE" \\
        -t "$TARGET_DURATION" \\
        -map 0:v -map 1:a \\
        -c:v copy -c:a aac -b:a 192k \\
        -shortest \\
        "${{OUTPUT_NAME}}.mp4" -y -loglevel warning
fi

# Cleanup
rm -f video_no_music.mp4 sequence.txt

# Results
if [[ -f "${{OUTPUT_NAME}}.mp4" ]]; then
    final_duration=$(ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${{OUTPUT_NAME}}.mp4" | cut -d. -f1)
    final_size=$(ls -lh "${{OUTPUT_NAME}}.mp4" | awk '{{print $5}}')
    
    echo ""
    echo "🎉 MONTAGE COMPLETE!"
    echo "📊 Output: ${{OUTPUT_NAME}}.mp4"
    echo "⏱️  Duration: ${{final_duration}}s"
    echo "🎞️  Content: $video_count videos, $image_count images"
    echo "📁 Size: $final_size"
    echo ""
    echo "✅ Settings applied:"
    echo "   🎵 Audio: $AUDIO_FILE"
    if [[ "$USE_AUDIO_TIMING" == "true" ]]; then
        echo "   ⏰ Custom timing: ${{AUDIO_START}}s to ${{AUDIO_END}}s"
    fi
    echo "   ⚡ Speed: ${{SPEED_MULTIPLIER}}x"
    echo "   🌈 Contrast: $CONTRAST, Saturation: $SATURATION"
    echo "   📸 Flash intensity: $IMAGE_FLASH_INTENSITY"
    echo "   🎲 Random shuffle: $RANDOM_SHUFFLE"
    echo "   ⚡ Glitch effects: $GLITCH_EFFECTS"
else
    echo "❌ Failed to create montage"
    exit 1
fi

echo "🧹 Cleanup: rm -rf clips"
'''
        
        return script