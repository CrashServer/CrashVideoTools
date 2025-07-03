"""
Parameter Tabs Component
Handles all parameter controls organized in tabs
"""

import tkinter as tk
from tkinter import ttk

class ParameterTabs:
    def __init__(self):
        # Parameters
        self.max_videos = tk.IntVar(value=25)
        self.max_images = tk.IntVar(value=10)
        self.video_duration = tk.DoubleVar(value=1.8)
        self.image_duration = tk.DoubleVar(value=0.3)
        self.speed_multiplier = tk.DoubleVar(value=2.0)
        self.video_quality = tk.IntVar(value=23)
        self.image_quality = tk.IntVar(value=18)
        self.contrast = tk.DoubleVar(value=1.2)
        self.saturation = tk.DoubleVar(value=1.3)
        self.brightness = tk.DoubleVar(value=0.0)
        self.match_audio_length = tk.BooleanVar(value=True)
        self.random_shuffle = tk.BooleanVar(value=True)
        self.glitch_effects = tk.BooleanVar(value=True)
        self.image_flash_intensity = tk.DoubleVar(value=3.0)
        
        # For logging (will be set by parent)
        self.log = None
        
    def setup_ui(self, parent, start_row):
        """Setup parameter tabs UI"""
        row = start_row
        
        # Parameters Notebook
        notebook = ttk.Notebook(parent)
        notebook.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        parent.rowconfigure(row, weight=1)
        row += 1
        
        # Basic Parameters Tab
        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="Basic Settings")
        self.setup_basic_tab(basic_frame)
        
        # Advanced Parameters Tab
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text="Advanced Effects")
        self.setup_advanced_tab(advanced_frame)
        
        # Output Settings Tab
        output_frame = ttk.Frame(notebook, padding="10")
        notebook.add(output_frame, text="Output Settings")
        self.setup_output_tab(output_frame)
        
        return row
        
    def setup_basic_tab(self, parent):
        """Setup basic parameters tab"""
        row = 0
        
        # File Limits
        ttk.Label(parent, text="📁 File Limits", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        ttk.Label(parent, text="Max Videos:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=5, to=50, variable=self.max_videos, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.max_videos).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        ttk.Label(parent, text="Max Images:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=1, to=20, variable=self.max_images, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.max_images).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Timing
        ttk.Label(parent, text="⏱️ Timing", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        row += 1
        
        ttk.Label(parent, text="Video Clip Duration (s):").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=0.5, to=5.0, variable=self.video_duration, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.video_duration).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        ttk.Label(parent, text="Image Flash Duration (s):").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=0.1, to=1.0, variable=self.image_duration, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.image_duration).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        ttk.Label(parent, text="Speed Multiplier:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=1.0, to=4.0, variable=self.speed_multiplier, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.speed_multiplier).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Quality
        ttk.Label(parent, text="🎥 Quality", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        row += 1
        
        ttk.Label(parent, text="Video Quality (CRF):").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=15, to=35, variable=self.video_quality, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.video_quality).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        ttk.Label(parent, text="Image Quality (CRF):").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=15, to=35, variable=self.image_quality, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.image_quality).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
    def setup_advanced_tab(self, parent):
        """Setup advanced effects tab"""
        row = 0
        
        # Color Effects
        ttk.Label(parent, text="🌈 Color Effects", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        ttk.Label(parent, text="Contrast:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=0.5, to=3.0, variable=self.contrast, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.contrast).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        ttk.Label(parent, text="Saturation:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=0.5, to=4.0, variable=self.saturation, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.saturation).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        ttk.Label(parent, text="Brightness:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=-0.5, to=0.5, variable=self.brightness, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.brightness).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Image Flash Effects
        ttk.Label(parent, text="📸 Image Flash Effects", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        row += 1
        
        ttk.Label(parent, text="Flash Intensity:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=1.0, to=5.0, variable=self.image_flash_intensity, orient=tk.HORIZONTAL, length=200).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(parent, textvariable=self.image_flash_intensity).grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Options
        ttk.Label(parent, text="⚡ Special Effects", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        row += 1
        
        ttk.Checkbutton(parent, text="Enable Glitch Effects", variable=self.glitch_effects).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Checkbutton(parent, text="Random Shuffle", variable=self.random_shuffle).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
    def setup_output_tab(self, parent):
        """Setup output settings tab"""
        row = 0
        
        ttk.Label(parent, text="📤 Output Settings", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        ttk.Checkbutton(parent, text="Match Audio Length", variable=self.match_audio_length).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        row += 1
        
        ttk.Label(parent, text="(When enabled, video duration will match audio file length exactly)", 
                 foreground="gray").grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 20))
        row += 1
        
        # Preset buttons
        ttk.Label(parent, text="🎛️ Presets", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        preset_frame = ttk.Frame(parent)
        preset_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W)
        
        ttk.Button(preset_frame, text="🚀 Fast & Dynamic", command=self.preset_fast).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(preset_frame, text="🎨 Artistic", command=self.preset_artistic).grid(row=0, column=1, padx=5)
        ttk.Button(preset_frame, text="⚡ Extreme Glitch", command=self.preset_glitch).grid(row=0, column=2, padx=5)
        ttk.Button(preset_frame, text="💎 High Quality", command=self.preset_quality).grid(row=0, column=3, padx=5)
        row += 1
        
    def preset_fast(self):
        """Fast & Dynamic preset"""
        self.speed_multiplier.set(3.0)
        self.video_duration.set(1.0)
        self.image_duration.set(0.2)
        self.contrast.set(1.4)
        self.saturation.set(1.6)
        self.glitch_effects.set(True)
        self.image_flash_intensity.set(4.0)
        if self.log:
            self.log("🚀 Applied Fast & Dynamic preset")
        
    def preset_artistic(self):
        """Artistic preset"""
        self.speed_multiplier.set(1.5)
        self.video_duration.set(2.5)
        self.image_duration.set(0.5)
        self.contrast.set(1.1)
        self.saturation.set(1.8)
        self.brightness.set(0.1)
        self.glitch_effects.set(False)
        self.image_flash_intensity.set(2.0)
        if self.log:
            self.log("🎨 Applied Artistic preset")
        
    def preset_glitch(self):
        """Extreme Glitch preset"""
        self.speed_multiplier.set(2.5)
        self.video_duration.set(0.8)
        self.image_duration.set(0.15)
        self.contrast.set(2.0)
        self.saturation.set(3.0)
        self.glitch_effects.set(True)
        self.image_flash_intensity.set(5.0)
        if self.log:
            self.log("⚡ Applied Extreme Glitch preset")
        
    def preset_quality(self):
        """High Quality preset"""
        self.speed_multiplier.set(1.8)
        self.video_duration.set(2.0)
        self.image_duration.set(0.4)
        self.video_quality.set(18)
        self.image_quality.set(15)
        self.contrast.set(1.15)
        self.saturation.set(1.25)
        self.glitch_effects.set(False)
        if self.log:
            self.log("💎 Applied High Quality preset")
        
    def set_log_callback(self, log_callback):
        """Set logging callback function"""
        self.log = log_callback
        
    def get_settings(self):
        """Get parameter settings as dictionary"""
        return {
            'max_videos': self.max_videos.get(),
            'max_images': self.max_images.get(),
            'video_duration': self.video_duration.get(),
            'image_duration': self.image_duration.get(),
            'speed_multiplier': self.speed_multiplier.get(),
            'video_quality': self.video_quality.get(),
            'image_quality': self.image_quality.get(),
            'contrast': self.contrast.get(),
            'saturation': self.saturation.get(),
            'brightness': self.brightness.get(),
            'match_audio_length': self.match_audio_length.get(),
            'random_shuffle': self.random_shuffle.get(),
            'glitch_effects': self.glitch_effects.get(),
            'image_flash_intensity': self.image_flash_intensity.get()
        }
        
    def apply_settings(self, settings):
        """Apply parameter settings from dictionary"""
        if 'max_videos' in settings:
            self.max_videos.set(settings['max_videos'])
        if 'max_images' in settings:
            self.max_images.set(settings['max_images'])
        if 'video_duration' in settings:
            self.video_duration.set(settings['video_duration'])
        if 'image_duration' in settings:
            self.image_duration.set(settings['image_duration'])
        if 'speed_multiplier' in settings:
            self.speed_multiplier.set(settings['speed_multiplier'])
        if 'video_quality' in settings:
            self.video_quality.set(settings['video_quality'])
        if 'image_quality' in settings:
            self.image_quality.set(settings['image_quality'])
        if 'contrast' in settings:
            self.contrast.set(settings['contrast'])
        if 'saturation' in settings:
            self.saturation.set(settings['saturation'])
        if 'brightness' in settings:
            self.brightness.set(settings['brightness'])
        if 'match_audio_length' in settings:
            self.match_audio_length.set(settings['match_audio_length'])
        if 'random_shuffle' in settings:
            self.random_shuffle.set(settings['random_shuffle'])
        if 'glitch_effects' in settings:
            self.glitch_effects.set(settings['glitch_effects'])
        if 'image_flash_intensity' in settings:
            self.image_flash_intensity.set(settings['image_flash_intensity'])
