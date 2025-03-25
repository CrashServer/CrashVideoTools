import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import logging
import queue
from PIL import Image, ImageTk
import cv2
import time
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import subprocess
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RealTimeDeepSceneDetector:
    def __init__(self, threshold=0.6, min_scene_length=1.0, batch_size=8, frame_skip=2, min_file_size=1.0, encoding_format='mp4', encoding_speed='medium', keyframe_interval=30, verbose=False):
        """
        Initialize the AI-based scene detector with a pre-trained model.

        Args:
            threshold: Similarity threshold for scene change detection (0-1)
            min_scene_length: Minimum length of a scene in seconds
            batch_size: Number of frames to process in a batch
            frame_skip: Number of frames to skip between processed frames
            min_file_size: Minimum file size in MB to keep
            encoding_format: Encoding format for output videos (e.g., 'mp4', 'avi')
            encoding_speed: Encoding speed (e.g., 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')
            keyframe_interval: Keyframe interval for output videos
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        if self.verbose:
            logger.info("Initializing AI scene detector...")
            logger.info(f"Using similarity threshold: {threshold}")
            logger.info(f"Using minimum scene length: {min_scene_length} seconds")
            logger.info(f"Using batch size: {batch_size}")
            logger.info(f"Using frame skip: {frame_skip}")
            logger.info(f"Using minimum file size: {min_file_size} MB")
            logger.info(f"Using encoding format: {encoding_format}")
            logger.info(f"Using encoding speed: {encoding_speed}")
            logger.info(f"Using keyframe interval: {keyframe_interval}")

        # Load pre-trained ResNet model
        start_time = time.time()
        if self.verbose:
            logger.info("Loading pre-trained ResNet-50 model...")

        self.model = models.resnet50(weights='IMAGENET1K_V1')
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_available = torch.cuda.is_available()

        # Always show GPU status clearly, regardless of verbose setting
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 GPU ENABLED: Using {gpu_name}")
        else:
            logger.info("⚠️ GPU NOT AVAILABLE: Using CPU only (processing will be slower)")

        if self.verbose:
            if gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU Memory: {gpu_memory:.2f} GB")

        self.model = self.model.to(self.device)

        load_time = time.time() - start_time
        if self.verbose:
            logger.info(f"Model loaded in {load_time:.2f} seconds")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.min_file_size = min_file_size
        self.encoding_format = encoding_format
        self.encoding_speed = encoding_speed
        self.keyframe_interval = keyframe_interval

    def extract_features(self, frames):
        """Extract deep features from a batch of frames using the model."""
        if self.verbose:
            logger.debug(f"Extracting features from {len(frames)} frames")

        images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        images = [self.transform(image).unsqueeze(0) for image in images]
        images = torch.cat(images).to(self.device)

        with torch.no_grad():
            features = self.model(images)

        return features.squeeze().cpu().numpy()

    def cosine_similarity(self, feat1, feat2):
        """Calculate cosine similarity between two feature vectors."""
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        if self.verbose:
            logger.debug(f"Frame similarity: {similarity:.4f}")
        return similarity

    def process_video_realtime(self, video_path, output_folder, start_time=0.0, extract_immediately=True, max_scene_length=20.0, ignore_small_files=False):
        """
        Process video in real-time, detecting and extracting scenes immediately.

        Args:
            video_path: Path to the input video
            output_folder: Folder to save extracted scenes
            start_time: Time to start processing from (in seconds)
            extract_immediately: Extract each scene as soon as it's detected
            max_scene_length: Maximum length of a scene in seconds (splits longer scenes)
            ignore_small_files: Ignore or remove files smaller than the specified minimum file size

        Returns:
            List of (start_time, end_time) tuples indicating scenes
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            if self.verbose:
                logger.info(f"Created output directory: {output_folder}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        if self.verbose:
            logger.info(f"Video information:")
            logger.info(f"  - Path: {video_path}")
            logger.info(f"  - FPS: {fps:.2f}")
            logger.info(f"  - Total frames: {total_frames}")
            logger.info(f"  - Duration: {duration:.2f} seconds")

        # Set the starting frame based on start_time
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        prev_features = None
        scenes = []
        scene_start_frame = start_frame
        scene_start_time = start_time

        # Process every n-th frame for efficiency
        sampling_rate = max(1, int(fps / 2))  # Sample at half the fps rate
        if self.verbose:
            logger.info(f"Sampling every {sampling_rate} frames (approx. {fps/sampling_rate:.2f} frames/second)")

        # Progress bar
        pbar = tqdm(total=total_frames // sampling_rate, desc="Processing video")

        frame_idx = start_frame
        scene_number = 1
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        frame_buffer = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every n-th frame
            if frame_idx % sampling_rate == 0:
                current_time = frame_idx / fps

                if self.verbose:
                    logger.debug(f"Processing frame {frame_idx} at {current_time:.2f}s")

                frame_buffer.append(frame)

                if len(frame_buffer) >= self.batch_size or (not ret and frame_buffer):
                    features = self.extract_features(frame_buffer)

                    for i, feat in enumerate(features):
                        if prev_features is not None:
                            similarity = self.cosine_similarity(prev_features, feat)

                            # Check if current scene is too long (> max_scene_length)
                            current_scene_duration = current_time - scene_start_time
                            force_split = current_scene_duration >= max_scene_length

                            # Split if similarity below threshold OR scene is too long
                            if similarity < self.threshold or force_split:
                                scene_end_frame = frame_idx - (len(features) - i - 1) * self.frame_skip
                                scene_end_time = scene_end_frame / fps

                                split_reason = "scene change detected" if similarity < self.threshold else "maximum length reached"

                                if self.verbose:
                                    logger.info(f"Scene split ({split_reason}) at frame {frame_idx}, time {current_time:.2f}s")
                                    logger.info(f"Scene {scene_number}: {scene_start_time:.2f}s to {scene_end_time:.2f}s (duration: {scene_end_time-scene_start_time:.2f}s)")
                                else:
                                    # Always show scene splits even in non-verbose mode
                                    print(f"Scene {scene_number}: {scene_start_time:.2f}s to {scene_end_time:.2f}s ({split_reason})")

                                scenes.append((scene_start_time, scene_end_time))

                                # Extract the scene immediately if requested
                                if extract_immediately:
                                    output_path = os.path.join(output_folder, f"{video_name}_scene_{scene_number:03d}.{self.encoding_format}")
                                    duration = scene_end_time - scene_start_time

                                    if self.verbose:
                                        logger.info(f"Extracting scene {scene_number} to {output_path}")

                                    extract_scene(video_path, scene_start_time, duration, output_path, self.verbose, self.encoding_speed, self.keyframe_interval)

                                    # Check file size and remove if smaller than the specified minimum file size
                                    if ignore_small_files and os.path.exists(output_path):
                                        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                                        if file_size < self.min_file_size:
                                            os.remove(output_path)
                                            if self.verbose:
                                                logger.info(f"Removed small file: {output_path} ({file_size:.2f} MB)")
                                            scene_number -= 1  # Decrement scene number if file is removed

                                scene_number += 1

                                scene_start_frame = frame_idx
                                scene_start_time = current_time

                        prev_features = feat
                    frame_buffer = []
                    pbar.update(len(features))

            frame_idx += self.frame_skip

        # Add the final scene
        final_time = frame_idx / fps
        if self.verbose:
            logger.info(f"Adding final scene: {scene_start_time:.2f}s to {final_time:.2f}s")

        scenes.append((scene_start_time, final_time))

        # Extract the final scene
        if extract_immediately:
            output_path = os.path.join(output_folder, f"{video_name}_scene_{scene_number:03d}.{self.encoding_format}")
            duration = final_time - scene_start_time
            extract_scene(video_path, scene_start_time, duration, output_path, self.verbose, self.encoding_speed, self.keyframe_interval)

            # Check file size and remove if smaller than the specified minimum file size
            if ignore_small_files and os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                if file_size < self.min_file_size:
                    os.remove(output_path)
                    if self.verbose:
                        logger.info(f"Removed small file: {output_path} ({file_size:.2f} MB)")

        cap.release()
        pbar.close()

        if self.verbose:
            logger.info(f"Completed video processing")
            logger.info(f"Detected {len(scenes)} scenes")

        return scenes

def extract_scene(video_path, start_time, duration, output_path, verbose=False, encoding_speed='medium', keyframe_interval=30):
    """Extract a single scene using FFmpeg."""
    if verbose:
        logger.info(f"Extracting scene: {start_time:.2f}s to {start_time + duration:.2f}s")

    # Use FFmpeg to extract the scene
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264', '-c:a', 'aac',
        '-preset', encoding_speed,
        '-g', str(keyframe_interval),
        '-strict', 'experimental',
        output_path
    ]

    if verbose:
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        subprocess.call(cmd)
    else:
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if verbose:
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Scene extracted successfully: {output_path} ({file_size:.2f} MB)")
        else:
            logger.warning(f"Failed to extract scene to {output_path}")

# Configure custom logging to redirect to our GUI
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

class SceneDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Scene Detector")
        self.root.geometry("1024x800")  # Reduced size since we're removing statistics panel
        self.root.minsize(1024, 800)

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_name = "None"
            self.gpu_memory = 0

        # Create a queue for logging
        self.log_queue = queue.Queue()
        self.setup_logging()

        # Variables
        self.video_path = tk.StringVar()
        self.output_folder = tk.StringVar(value=os.path.join(os.getcwd(), "ai_extracted_scenes"))
        self.threshold = tk.DoubleVar(value=0.6)
        self.max_length = tk.DoubleVar(value=20.0)
        self.min_length = tk.DoubleVar(value=1.0)
        self.batch_size = tk.IntVar(value=8)
        self.frame_skip = tk.IntVar(value=2)
        self.min_file_size = tk.DoubleVar(value=1.0)
        self.encoding_format = tk.StringVar(value='mp4')
        self.encoding_speed = tk.StringVar(value='medium')
        self.keyframe_interval = tk.IntVar(value=30)
        self.use_gpu = tk.BooleanVar(value=self.gpu_available)
        self.extract_immediately = tk.BooleanVar(value=True)
        self.verbose = tk.BooleanVar(value=False)
        self.ignore_small_files = tk.BooleanVar(value=False)
        self.start_time = tk.DoubleVar(value=0.0)
        self.video_duration = 0.0  # Track video duration

        # Processing variables
        self.is_processing = False
        self.processing_thread = None
        self.preview_cap = None
        self.preview_thread = None
        self.stop_preview = False
        self.files_created = 0

        # Create UI
        self.create_widgets()

        # Position all frames
        self.layout_widgets()

        # Setup periodic log reader
        self.root.after(100, self.check_logs)

    def setup_logging(self):
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create our custom handler
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        queue_handler.setFormatter(formatter)
        root_logger.addHandler(queue_handler)

    def create_widgets(self):
        # Main frames
        self.create_gpu_frame()
        self.create_input_frame()
        self.create_options_frame()
        self.create_preview_frame()
        self.create_log_frame()
        self.create_buttons_frame()

    def layout_widgets(self):
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(3, weight=1)  # Preview gets most vertical space
        self.root.grid_rowconfigure(4, weight=1)  # Log frame gets remaining space

        # Place frames
        self.gpu_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.input_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.options_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.preview_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        self.log_frame.grid(row=4, column=0, padx=10, pady=5, sticky="nsew")
        self.buttons_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")

    def create_gpu_frame(self):
        self.gpu_frame = ttk.LabelFrame(self.root, text="GPU Status")

        # Configure grid
        self.gpu_frame.grid_columnconfigure(1, weight=1)

        # GPU status indicator (colored box)
        indicator_frame = ttk.Frame(self.gpu_frame, width=20, height=20)
        indicator_frame.grid(row=0, column=0, padx=10, pady=5)

        indicator_color = "#22CC22" if self.gpu_available else "#CC2222"  # Green if available, red if not
        self.gpu_indicator = tk.Canvas(indicator_frame, width=20, height=20, bg=indicator_color,
                                       highlightthickness=0)
        self.gpu_indicator.pack(fill=tk.BOTH, expand=True)

        # GPU status text
        if self.gpu_available:
            status_text = f"GPU ENABLED: {self.gpu_name} ({self.gpu_memory:.2f} GB)"
            self.use_gpu.set(True)  # Default to using GPU if available
        else:
            status_text = "GPU NOT AVAILABLE: Using CPU only (processing will be slower)"
            self.use_gpu.set(False)  # Can't use GPU if not available

        gpu_label = ttk.Label(self.gpu_frame, text=status_text, font=("Arial", 10, "bold"))
        gpu_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    def create_input_frame(self):
        self.input_frame = ttk.LabelFrame(self.root, text="Input")

        # Configure grid
        self.input_frame.grid_columnconfigure(1, weight=1)

        # Video path
        ttk.Label(self.input_frame, text="Video Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        video_entry = ttk.Entry(self.input_frame, textvariable=self.video_path)
        video_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.input_frame, text="Browse", command=self.browse_video).grid(row=0, column=2, padx=5, pady=5)

        # Output folder
        ttk.Label(self.input_frame, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        output_entry = ttk.Entry(self.input_frame, textvariable=self.output_folder)
        output_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.input_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

    def browse_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")])
        if video_path:
            self.video_path.set(video_path)

    def browse_output(self):
        output_folder = filedialog.askdirectory()
        if output_folder:
            self.output_folder.set(output_folder)

    def create_options_frame(self):
        self.options_frame = ttk.LabelFrame(self.root, text="Options")

        # Configure grid
        self.options_frame.grid_columnconfigure(1, weight=1)

        # Threshold
        ttk.Label(self.options_frame, text="Similarity Threshold:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        threshold_entry = ttk.Entry(self.options_frame, textvariable=self.threshold)
        threshold_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Max scene length
        ttk.Label(self.options_frame, text="Max Scene Length (s):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        max_length_entry = ttk.Entry(self.options_frame, textvariable=self.max_length)
        max_length_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Min scene length
        ttk.Label(self.options_frame, text="Min Scene Length (s):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        min_length_entry = ttk.Entry(self.options_frame, textvariable=self.min_length)
        min_length_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Batch size
        ttk.Label(self.options_frame, text="Batch Size:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        batch_size_entry = ttk.Entry(self.options_frame, textvariable=self.batch_size)
        batch_size_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Frame skip
        ttk.Label(self.options_frame, text="Frame Skip:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        frame_skip_entry = ttk.Entry(self.options_frame, textvariable=self.frame_skip)
        frame_skip_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        # Min file size
        ttk.Label(self.options_frame, text="Min File Size (MB):").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        min_file_size_entry = ttk.Entry(self.options_frame, textvariable=self.min_file_size)
        min_file_size_entry.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        # Encoding format
        ttk.Label(self.options_frame, text="Encoding Format:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        encoding_format_entry = ttk.Entry(self.options_frame, textvariable=self.encoding_format)
        encoding_format_entry.grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        # Encoding speed
        ttk.Label(self.options_frame, text="Encoding Speed:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        encoding_speed_entry = ttk.Entry(self.options_frame, textvariable=self.encoding_speed)
        encoding_speed_entry.grid(row=7, column=1, padx=5, pady=5, sticky="ew")

        # Keyframe interval
        ttk.Label(self.options_frame, text="Keyframe Interval:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        keyframe_interval_entry = ttk.Entry(self.options_frame, textvariable=self.keyframe_interval)
        keyframe_interval_entry.grid(row=8, column=1, padx=5, pady=5, sticky="ew")

        # Use GPU
        ttk.Checkbutton(self.options_frame, text="Use GPU", variable=self.use_gpu).grid(row=9, column=0, padx=5, pady=5, sticky="w")

        # Extract immediately
        ttk.Checkbutton(self.options_frame, text="Extract Immediately", variable=self.extract_immediately).grid(row=10, column=0, padx=5, pady=5, sticky="w")

        # Verbose logging
        ttk.Checkbutton(self.options_frame, text="Verbose Logging", variable=self.verbose).grid(row=11, column=0, padx=5, pady=5, sticky="w")

        # Ignore small files
        ttk.Checkbutton(self.options_frame, text="Ignore Small Files", variable=self.ignore_small_files).grid(row=12, column=0, padx=5, pady=5, sticky="w")

    def create_preview_frame(self):
        self.preview_frame = ttk.LabelFrame(self.root, text="Video Preview")

        # Configure grid
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)

        # Video preview canvas
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="black")
        self.preview_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    def create_log_frame(self):
        self.log_frame = ttk.LabelFrame(self.root, text="Log")

        # Configure grid
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(0, weight=1)

        # Log text box
        self.log_text = scrolledtext.ScrolledText(self.log_frame, state='disabled', height=10)
        self.log_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    def create_buttons_frame(self):
        self.buttons_frame = ttk.Frame(self.root)

        # Configure grid
        self.buttons_frame.grid_columnconfigure(0, weight=1)
        self.buttons_frame.grid_columnconfigure(1, weight=1)

        # Start button
        self.start_button = ttk.Button(self.buttons_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Stop button
        self.stop_button = ttk.Button(self.buttons_frame, text="Stop Processing", command=self.stop_processing, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def check_logs(self):
        try:
            while True:
                record = self.log_queue.get_nowait()
                self.log_text.config(state='normal')
                self.log_text.insert(tk.END, self.log_queue.get().message + '\n')
                self.log_text.config(state='disabled')
                self.log_text.yview(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.check_logs)

    def start_processing(self):
        if self.is_processing:
            return

        self.is_processing = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')

        video_path = self.video_path.get()
        output_folder = self.output_folder.get()
        threshold = self.threshold.get()
        max_length = self.max_length.get()
        min_length = self.min_length.get()
        batch_size = self.batch_size.get()
        frame_skip = self.frame_skip.get()
        min_file_size = self.min_file_size.get()
        encoding_format = self.encoding_format.get()
        encoding_speed = self.encoding_speed.get()
        keyframe_interval = self.keyframe_interval.get()
        use_gpu = self.use_gpu.get()
        extract_immediately = self.extract_immediately.get()
        verbose = self.verbose.get()
        ignore_small_files = self.ignore_small_files.get()
        start_time = self.start_time.get()

        self.processing_thread = threading.Thread(target=self.process_video, args=(
            video_path, output_folder, threshold, max_length, min_length, batch_size, frame_skip, min_file_size,
            encoding_format, encoding_speed, keyframe_interval, use_gpu, extract_immediately, verbose, ignore_small_files,
            start_time))
        self.processing_thread.start()

    def process_video(self, video_path, output_folder, threshold, max_length, min_length, batch_size, frame_skip,
                      min_file_size, encoding_format, encoding_speed, keyframe_interval, use_gpu, extract_immediately,
                      verbose, ignore_small_files, start_time):
        detector = RealTimeDeepSceneDetector(
            threshold=threshold,
            min_scene_length=min_length,
            batch_size=batch_size,
            frame_skip=frame_skip,
            min_file_size=min_file_size,
            encoding_format=encoding_format,
            encoding_speed=encoding_speed,
            keyframe_interval=keyframe_interval,
            verbose=verbose
        )

        if not use_gpu:
            detector.model = detector.model.cpu()
            detector.device = torch.device("cpu")

        scenes = detector.process_video_realtime(
            video_path, output_folder, start_time=start_time, extract_immediately=extract_immediately,
            max_scene_length=max_length, ignore_small_files=ignore_small_files
        )

        self.files_created = len(scenes)
        self.is_processing = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def stop_processing(self):
        self.is_processing = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

        if self.processing_thread is not None:
            self.processing_thread.join()

if __name__ == "__main__":
    root = tk.Tk()
    app = SceneDetectorApp(root)
    root.mainloop()
