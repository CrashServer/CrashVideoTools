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
    def __init__(self, threshold=0.6, verbose=False):
        """
        Initialize the AI-based scene detector with a pre-trained model.
        
        Args:
            threshold: Similarity threshold for scene change detection (0-1)
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        if self.verbose:
            logger.info("Initializing AI scene detector...")
            logger.info(f"Using similarity threshold: {threshold}")
        
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
    
    def extract_features(self, frame):
        """Extract deep features from a frame using the model."""
        if self.verbose:
            logger.debug("Extracting features from frame")
            
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image)
        
        return features.squeeze().cpu().numpy()
    
    def cosine_similarity(self, feat1, feat2):
        """Calculate cosine similarity between two feature vectors."""
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        if self.verbose:
            logger.debug(f"Frame similarity: {similarity:.4f}")
        return similarity
    
    def process_video_realtime(self, video_path, output_folder, extract_immediately=True, max_scene_length=20.0):
        """
        Process video in real-time, detecting and extracting scenes immediately.
        
        Args:
            video_path: Path to the input video
            output_folder: Folder to save extracted scenes
            extract_immediately: Extract each scene as soon as it's detected
            max_scene_length: Maximum length of a scene in seconds (splits longer scenes)
            
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
        
        prev_features = None
        scenes = []
        scene_start_frame = 0
        scene_start_time = 0
        
        # Process every n-th frame for efficiency
        sampling_rate = max(1, int(fps / 2))  # Sample at half the fps rate
        if self.verbose:
            logger.info(f"Sampling every {sampling_rate} frames (approx. {fps/sampling_rate:.2f} frames/second)")
        
        # Progress bar
        pbar = tqdm(total=total_frames // sampling_rate, desc="Processing video")
        
        frame_idx = 0
        scene_number = 1
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every n-th frame
            if frame_idx % sampling_rate == 0:
                current_time = frame_idx / fps
                
                if self.verbose:
                    logger.debug(f"Processing frame {frame_idx} at {current_time:.2f}s")
                
                features = self.extract_features(frame)
                
                if prev_features is not None:
                    similarity = self.cosine_similarity(prev_features, features)
                    
                    # Check if current scene is too long (> max_scene_length)
                    current_scene_duration = current_time - scene_start_time
                    force_split = current_scene_duration >= max_scene_length
                    
                    # Split if similarity below threshold OR scene is too long
                    if similarity < self.threshold or force_split:
                        scene_end_frame = frame_idx - 1
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
                            output_path = os.path.join(output_folder, f"{video_name}_scene_{scene_number:03d}.mp4")
                            duration = scene_end_time - scene_start_time
                            
                            if self.verbose:
                                logger.info(f"Extracting scene {scene_number} to {output_path}")
                            
                            extract_scene(video_path, scene_start_time, duration, output_path, self.verbose)
                            scene_number += 1
                        
                        scene_start_frame = frame_idx
                        scene_start_time = current_time
                
                prev_features = features
                pbar.update(1)
            
            frame_idx += 1
        
        # Add the final scene
        final_time = frame_idx / fps
        if self.verbose:
            logger.info(f"Adding final scene: {scene_start_time:.2f}s to {final_time:.2f}s")
        
        scenes.append((scene_start_time, final_time))
        
        # Extract the final scene
        if extract_immediately:
            output_path = os.path.join(output_folder, f"{video_name}_scene_{scene_number:03d}.mp4")
            duration = final_time - scene_start_time
            extract_scene(video_path, scene_start_time, duration, output_path, self.verbose)
        
        cap.release()
        pbar.close()
        
        if self.verbose:
            logger.info(f"Completed video processing")
            logger.info(f"Detected {len(scenes)} scenes")
        
        return scenes

def extract_scene(video_path, start_time, duration, output_path, verbose=False):
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
        self.root.geometry("900x750")  # Increased height for GPU status
        self.root.minsize(900, 750)
        
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
        self.use_gpu = tk.BooleanVar(value=self.gpu_available)
        self.extract_immediately = tk.BooleanVar(value=True)
        self.verbose = tk.BooleanVar(value=False)
        
        # Processing variables
        self.is_processing = False
        self.processing_thread = None
        self.preview_cap = None
        self.preview_thread = None
        self.stop_preview = False
        
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
        self.input_frame = ttk.LabelFrame(self.root, text="Input/Output")
        
        # Configure grid
        self.input_frame.grid_columnconfigure(1, weight=1)
        
        # Video input
        ttk.Label(self.input_frame, text="Video File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        video_entry = ttk.Entry(self.input_frame, textvariable=self.video_path)
        video_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.input_frame, text="Browse", command=self.browse_video).grid(row=0, column=2, padx=5, pady=5)
        
        # Output folder
        ttk.Label(self.input_frame, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        output_entry = ttk.Entry(self.input_frame, textvariable=self.output_folder)
        output_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.input_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

    def create_options_frame(self):
        self.options_frame = ttk.LabelFrame(self.root, text="Detection Options")
        
        # Configure grid - 2 columns for options
        self.options_frame.grid_columnconfigure(0, weight=1)
        self.options_frame.grid_columnconfigure(1, weight=1)
        
        # Left column options
        left_frame = ttk.Frame(self.options_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        left_frame.grid_columnconfigure(1, weight=1)
        
        # Threshold
        ttk.Label(left_frame, text="Similarity Threshold:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        threshold_scale = ttk.Scale(left_frame, from_=0.1, to=0.9, variable=self.threshold, orient="horizontal")
        threshold_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Threshold value display that updates with the scale
        self.threshold_value_label = ttk.Label(left_frame, text=f"{self.threshold.get():.1f}")
        self.threshold_value_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Update the threshold value label when the scale changes
        threshold_scale.bind("<Motion>", self.update_threshold_label)
        threshold_scale.bind("<ButtonRelease-1>", self.update_threshold_label)
        
        # Threshold explanation
        threshold_explanation = "Lower values (0.1-0.3): More sensitive, detects subtle changes\n" \
                               "Medium values (0.4-0.6): Balanced detection\n" \
                               "Higher values (0.7-0.9): Less sensitive, only major scene changes"
        ttk.Label(left_frame, text=threshold_explanation, font=("Arial", 8), foreground="dark gray").grid(
            row=1, column=0, columnspan=3, padx=5, pady=0, sticky="w")
        
        # Max scene length
        ttk.Label(left_frame, text="Max Scene Length (s):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        max_length_spin = ttk.Spinbox(left_frame, from_=5, to=120, textvariable=self.max_length)
        max_length_spin.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Right column options
        right_frame = ttk.Frame(self.options_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Checkboxes
        # Use GPU checkbox - enabled only if GPU is available
        gpu_checkbutton = ttk.Checkbutton(right_frame, text="Use GPU (if available)", variable=self.use_gpu)
        if not self.gpu_available:
            gpu_checkbutton.state(['disabled'])
        gpu_checkbutton.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Checkbutton(right_frame, text="Extract Immediately", variable=self.extract_immediately).grid(
            row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(right_frame, text="Verbose Logging", variable=self.verbose).grid(
            row=2, column=0, padx=5, pady=5, sticky="w")

    def update_threshold_label(self, event=None):
        self.threshold_value_label.config(text=f"{self.threshold.get():.1f}")

    def create_preview_frame(self):
        self.preview_frame = ttk.LabelFrame(self.root, text="Video Preview")
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        
        # Add canvas for video preview
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="black")
        self.preview_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Default preview message
        self.preview_canvas.create_text(
            self.preview_canvas.winfo_reqwidth() // 2, 
            self.preview_canvas.winfo_reqheight() // 2,
            text="Select a video file to preview", 
            fill="white", 
            font=("Arial", 12)
        )

    def create_log_frame(self):
        self.log_frame = ttk.LabelFrame(self.root, text="Processing Log")
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(0, weight=1)
        
        # Add scrolled text for logs
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

    def create_buttons_frame(self):
        self.buttons_frame = ttk.Frame(self.root)
        
        # Configure with equal weights
        self.buttons_frame.grid_columnconfigure(0, weight=1)
        self.buttons_frame.grid_columnconfigure(1, weight=1)
        
        # Create a frame for the buttons to center them
        button_container = ttk.Frame(self.buttons_frame)
        button_container.grid(row=0, column=0, columnspan=2)
        
        # Add processing buttons
        self.start_button = ttk.Button(button_container, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=10, pady=5)
        
        self.stop_button = ttk.Button(button_container, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.buttons_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
        )
        if filename:
            self.video_path.set(filename)
            self.start_preview()
            # Auto-create output folder name based on video name
            video_name = os.path.splitext(os.path.basename(filename))[0]
            self.output_folder.set(os.path.join(os.getcwd(), f"{video_name}_scenes"))

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def check_logs(self):
        # Process any new log records
        while not self.log_queue.empty():
            record = self.log_queue.get()
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, self.format_log(record) + '\n')
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        # Schedule the next check
        self.root.after(100, self.check_logs)

    def format_log(self, record):
        if record.levelname == 'INFO':
            return f"[INFO] {record.message}"
        elif record.levelname == 'WARNING':
            return f"[WARN] {record.message}"
        elif record.levelname == 'ERROR':
            return f"[ERROR] {record.message}"
        return f"[{record.levelname}] {record.message}"

    def start_processing(self):
        if not self.video_path.get():
            self.show_message("Error", "Please select a video file first.")
            return
            
        if not os.path.exists(self.video_path.get()):
            self.show_message("Error", f"Video file not found: {self.video_path.get()}")
            return
            
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder.get(), exist_ok=True)
            
        # Update UI state
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start(10)
        
        # Clear log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Set logging level based on verbose checkbox
        if self.verbose.get():
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.run_detector)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def run_detector(self):
        try:
            # Force CPU if not using GPU
            if not self.use_gpu.get():
                import torch
                torch.cuda.is_available = lambda: False
                
            # Initialize detector
            detector = RealTimeDeepSceneDetector(
                threshold=self.threshold.get(),
                verbose=self.verbose.get()
            )
            
            # Process video
            scenes = detector.process_video_realtime(
                self.video_path.get(),
                self.output_folder.get(),
                extract_immediately=self.extract_immediately.get(),
                max_scene_length=self.max_length.get()
            )
            
            # Log completion
            logging.info(f"Processing complete. Detected {len(scenes)} scenes.")
            logging.info(f"All scenes extracted to: {os.path.abspath(self.output_folder.get())}")
            
            # Show completion message
            self.root.after(0, lambda: self.show_message("Processing Complete", 
                f"Detected {len(scenes)} scenes.\nFiles saved to {os.path.abspath(self.output_folder.get())}"))
            
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            self.root.after(0, lambda: self.show_message("Error", f"Processing failed: {str(e)}"))
        finally:
            # Reset UI state
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()

    def stop_processing(self):
        if not self.is_processing:
            return
            
        logging.info("Stopping processing...")
        self.is_processing = False
        # The processing thread will eventually complete its current task and then exit
        self.reset_ui()

    def start_preview(self):
        # Stop any existing preview
        self.stop_preview = True
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=1.0)
        
        if not self.video_path.get() or not os.path.exists(self.video_path.get()):
            return
            
        # Reset stop flag
        self.stop_preview = False
        
        # Start new preview thread
        self.preview_thread = threading.Thread(target=self.video_preview_loop)
        self.preview_thread.daemon = True
        self.preview_thread.start()

    def video_preview_loop(self):
        try:
            # Open video file
            cap = cv2.VideoCapture(self.video_path.get())
            if not cap.isOpened():
                return
                
            # Get video info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate display size
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Update every 100ms (approximately 10 fps)
            interval = max(1, int(fps / 10))
            frame_count = 0
            
            while not self.stop_preview:
                ret, frame = cap.read()
                if not ret:
                    # Loop back to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                # Process every n-th frame
                if frame_count % interval == 0:
                    # Calculate scaling to fit in canvas
                    canvas_width = self.preview_canvas.winfo_width()
                    canvas_height = self.preview_canvas.winfo_height()
                    
                    if canvas_width <= 1 or canvas_height <= 1:
                        # Canvas not properly sized yet
                        time.sleep(0.1)
                        continue
                        
                    # Calculate aspect ratio-preserving scaling
                    scale = min(canvas_width / width, canvas_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # Resize frame
                    resized = cv2.resize(frame, (new_width, new_height))
                    # Convert from BGR to RGB
                    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    # Convert to PhotoImage
                    image = Image.fromarray(rgb_image)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    # Update canvas
                    self.preview_canvas.delete("all")
                    self.preview_canvas.create_image(
                        canvas_width // 2,
                        canvas_height // 2,
                        image=photo,
                        anchor="center"
                    )
                    self.preview_canvas.image = photo  # Keep a reference
                    
                    # Sleep to achieve desired frame rate
                    time.sleep(0.1)
                
                frame_count += 1
        except Exception as e:
            logging.error(f"Error in preview: {str(e)}")
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()

    def show_message(self, title, message):
        messagebox.showinfo(title, message)

if __name__ == "__main__":
    root = tk.Tk()
    app = SceneDetectorApp(root)
    root.mainloop()
