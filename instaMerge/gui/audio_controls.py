"""
Audio Controls Component
Handles audio file selection, timing controls, and audio analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog
import subprocess
import os
import glob

class AudioControls:
    def __init__(self):
        # Audio variables
        self.audio_file = tk.StringVar()
        self.audio_start_time = tk.DoubleVar(value=0.0)
        self.audio_end_time = tk.DoubleVar(value=0.0)
        self.audio_duration = tk.DoubleVar(value=0.0)
        self.use_audio_timing = tk.BooleanVar(value=False)
        
        # UI elements (will be set during setup)
        self.start_scale = None
        self.end_scale = None
        self.start_label = None
        self.end_label = None
        self.duration_label = None
        
    def setup_ui(self, parent, start_row, log_callback):
        """Setup audio controls UI"""
        self.log = log_callback
        row = start_row
        
        # Audio File Section
        ttk.Label(parent, text="🎵 Audio Source:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        audio_frame = ttk.Frame(parent)
        audio_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        audio_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(audio_frame, textvariable=self.audio_file, width=60).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(audio_frame, text="Browse", command=self.browse_audio).grid(row=0, column=1)
        ttk.Button(audio_frame, text="Auto-find", command=lambda: self.auto_find_audio(None, self.log)).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(audio_frame, text="Analyze", command=self.analyze_audio).grid(row=0, column=3, padx=(5, 0))
        row += 1
        
        # Audio timing controls
        timing_frame = ttk.LabelFrame(parent, text="🎵 Audio Timing", padding="5")
        timing_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        timing_frame.columnconfigure(1, weight=1)
        timing_frame.columnconfigure(4, weight=1)
        
        ttk.Checkbutton(timing_frame, text="Use custom audio timing", 
                       variable=self.use_audio_timing, 
                       command=self.toggle_audio_timing).grid(row=0, column=0, columnspan=6, sticky=tk.W, pady=(0, 5))
        
        # Start time
        ttk.Label(timing_frame, text="Start:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.start_scale = ttk.Scale(timing_frame, from_=0, to=300, variable=self.audio_start_time, 
                                    orient=tk.HORIZONTAL, length=150, command=self.update_audio_timing)
        self.start_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.start_label = ttk.Label(timing_frame, text="00:00")
        self.start_label.grid(row=1, column=2, sticky=tk.W, padx=(5, 10))
        
        # End time  
        ttk.Label(timing_frame, text="End:").grid(row=1, column=3, sticky=tk.W, padx=(0, 5))
        self.end_scale = ttk.Scale(timing_frame, from_=0, to=300, variable=self.audio_end_time, 
                                  orient=tk.HORIZONTAL, length=150, command=self.update_audio_timing)
        self.end_scale.grid(row=1, column=4, sticky=(tk.W, tk.E), padx=5)
        self.end_label = ttk.Label(timing_frame, text="00:00")
        self.end_label.grid(row=1, column=5, sticky=tk.W, padx=5)
        
        # Duration display
        self.duration_label = ttk.Label(timing_frame, text="Duration: 00:00", font=("Arial", 10, "bold"))
        self.duration_label.grid(row=2, column=0, columnspan=6, pady=(5, 0))
        
        # Quick timing buttons
        quick_frame = ttk.Frame(timing_frame)
        quick_frame.grid(row=3, column=0, columnspan=6, pady=(5, 0))
        
        ttk.Button(quick_frame, text="Full Track", command=self.set_full_audio).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(quick_frame, text="First 30s", command=lambda: self.set_quick_timing(0, 30)).grid(row=0, column=1, padx=5)
        ttk.Button(quick_frame, text="First 60s", command=lambda: self.set_quick_timing(0, 60)).grid(row=0, column=2, padx=5)
        ttk.Button(quick_frame, text="Last 30s", command=self.set_last_30s).grid(row=0, column=3, padx=5)
        
        # Initially disable timing controls
        self.toggle_audio_timing()
        row += 1
        
        return row
        
    def browse_audio(self):
        """Browse for audio file"""
        file_types = [
            ("All Audio/Video", "*.mp4 *.mov *.mkv *.avi *.wav *.mp3 *.aac *.flac *.ogg"),
            ("Video files", "*.mp4 *.mov *.mkv *.avi"),
            ("Audio files", "*.wav *.mp3 *.aac *.flac *.ogg"),
            ("All files", "*.*")
        ]
        audio_file = filedialog.askopenfilename(filetypes=file_types)
        if audio_file:
            self.audio_file.set(audio_file)
            self.analyze_audio()
            
    def auto_find_audio(self, directory=None, log_callback=None):
        """Auto-find 01.mp4 or first audio file in directory"""
        if not directory:
            # This will be set by the main GUI
            return
            
        if log_callback:
            self.log = log_callback
            
        # Look for 01.mp4 first
        target_file = os.path.join(directory, "01.mp4")
        if os.path.exists(target_file):
            self.audio_file.set(target_file)
            self.log("🎵 Found 01.mp4 as audio source")
            self.analyze_audio()
            return
            
        # Look for any audio/video file
        audio_extensions = ["*.mp4", "*.mov", "*.mkv", "*.avi", "*.wav", "*.mp3", "*.aac", "*.flac", "*.ogg"]
        for ext in audio_extensions:
            files = glob.glob(os.path.join(directory, ext))
            if files:
                self.audio_file.set(files[0])
                self.log(f"🎵 Auto-selected: {os.path.basename(files[0])}")
                self.analyze_audio()
                return
                
        self.log("⚠️ No audio files found in directory")
        
    def analyze_audio(self):
        """Analyze selected audio file and update timing controls"""
        audio_file = self.audio_file.get()
        if not audio_file or not os.path.exists(audio_file):
            return
            
        try:
            # Get audio duration using ffprobe
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                self.audio_duration.set(duration)
                
                # Update timing controls
                self.start_scale.config(to=duration)
                self.end_scale.config(to=duration)
                self.audio_end_time.set(duration)
                
                # Update display
                self.update_audio_timing()
                
                # Show info
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                file_format = os.path.splitext(audio_file)[1].lower()
                self.log(f"🎵 Analyzed: {minutes:02d}:{seconds:02d} ({file_format} format)")
                
            else:
                self.log("⚠️ Could not analyze audio file")
                
        except subprocess.TimeoutExpired:
            self.log("⚠️ Audio analysis timeout")
        except Exception as e:
            self.log(f"⚠️ Audio analysis error: {e}")
            
    def toggle_audio_timing(self):
        """Enable/disable audio timing controls"""
        state = "normal" if self.use_audio_timing.get() else "disabled"
        
        if self.start_scale:
            self.start_scale.config(state=state)
        if self.end_scale:
            self.end_scale.config(state=state)
        
        # Update timing display
        if self.use_audio_timing.get():
            self.update_audio_timing()
        else:
            if self.start_label:
                self.start_label.config(text="00:00")
            if self.end_label:
                self.end_label.config(text="00:00")
            if self.duration_label:
                self.duration_label.config(text="Duration: Full track")
                
    def update_audio_timing(self, *args):
        """Update audio timing displays"""
        if not self.use_audio_timing.get():
            return
            
        start_time = self.audio_start_time.get()
        end_time = self.audio_end_time.get()
        
        # Ensure end time is after start time
        if end_time <= start_time:
            end_time = start_time + 1
            self.audio_end_time.set(end_time)
        
        # Format time displays
        def format_time(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"
            
        if self.start_label:
            self.start_label.config(text=format_time(start_time))
        if self.end_label:
            self.end_label.config(text=format_time(end_time))
        
        duration = end_time - start_time
        if self.duration_label:
            self.duration_label.config(text=f"Duration: {format_time(duration)}")
        
    def set_full_audio(self):
        """Set timing to use full audio track"""
        self.audio_start_time.set(0)
        self.audio_end_time.set(self.audio_duration.get())
        self.update_audio_timing()
        
    def set_quick_timing(self, start, duration):
        """Set quick timing presets"""
        self.audio_start_time.set(start)
        end_time = min(start + duration, self.audio_duration.get())
        self.audio_end_time.set(end_time)
        self.update_audio_timing()
        
    def set_last_30s(self):
        """Set timing to last 30 seconds"""
        total_duration = self.audio_duration.get()
        start_time = max(0, total_duration - 30)
        self.audio_start_time.set(start_time)
        self.audio_end_time.set(total_duration)
        self.update_audio_timing()
        
    def validate_audio(self):
        """Validate audio file selection"""
        audio_file = self.audio_file.get()
        return audio_file and os.path.exists(audio_file)
        
    def get_settings(self):
        """Get audio settings as dictionary"""
        return {
            'audio_file': self.audio_file.get(),
            'audio_start_time': self.audio_start_time.get(),
            'audio_end_time': self.audio_end_time.get(),
            'use_audio_timing': self.use_audio_timing.get()
        }
        
    def apply_settings(self, settings):
        """Apply audio settings from dictionary"""
        if 'audio_file' in settings:
            self.audio_file.set(settings['audio_file'])
        if 'audio_start_time' in settings:
            self.audio_start_time.set(settings['audio_start_time'])
        if 'audio_end_time' in settings:
            self.audio_end_time.set(settings['audio_end_time'])
        if 'use_audio_timing' in settings:
            self.use_audio_timing.set(settings['use_audio_timing'])
            self.toggle_audio_timing()