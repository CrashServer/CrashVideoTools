"""
Main GUI Class for Video Montage Creator
Handles the main window layout and coordination between components
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from .audio_controls import AudioControls
from .parameter_tabs import ParameterTabs
from core.montage_processor import MontageProcessor
from utils.file_utils import FileUtils
from utils.settings_manager import SettingsManager

class VideoMontageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Montage Creator")
        self.root.geometry("800x900")
        
        # Initialize components
        self.file_utils = FileUtils()
        self.settings_manager = SettingsManager()
        self.montage_processor = MontageProcessor()
        
        # Variables
        self.working_dir = tk.StringVar(value=os.getcwd())
        self.output_name = tk.StringVar(value="MONTAGE")
        
        # Initialize sub-components
        self.audio_controls = AudioControls()
        self.parameter_tabs = ParameterTabs()
        
        self.setup_ui()
        self.update_file_counts()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Title
        title_label = ttk.Label(main_frame, text="🎬 Video Montage Creator", font=("Arial", 16, "bold"))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # Working Directory Section
        row = self._setup_directory_section(main_frame, row)
        
        # Audio Controls Section
        row = self.audio_controls.setup_ui(main_frame, row, self.log)
        
        # Parameter Tabs Section
        row = self.parameter_tabs.setup_ui(main_frame, row)
        
        # Control Buttons
        row = self._setup_control_buttons(main_frame, row)
        
        # Progress and Log Section
        self._setup_progress_log(main_frame, row)
        
    def _setup_directory_section(self, parent, start_row):
        """Setup directory selection section"""
        row = start_row
        
        ttk.Label(parent, text="📁 Working Directory:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        dir_frame = ttk.Frame(parent)
        dir_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        dir_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(dir_frame, textvariable=self.working_dir, width=60).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).grid(row=0, column=1)
        ttk.Button(dir_frame, text="Refresh", command=self.update_file_counts).grid(row=0, column=2, padx=(5, 0))
        row += 1
        
        # File counts
        self.file_counts_label = ttk.Label(parent, text="", foreground="blue")
        self.file_counts_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        return row
        
    def _setup_control_buttons(self, parent, start_row):
        """Setup control buttons section"""
        row = start_row
        
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=row, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(control_frame, text="💾 Save Settings", command=self.save_settings).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(control_frame, text="📁 Load Settings", command=self.load_settings).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="🎬 Create Montage", command=self.create_montage, style="Accent.TButton").grid(row=0, column=2, padx=(10, 0))
        row += 1
        
        return row
        
    def _setup_progress_log(self, parent, start_row):
        """Setup progress bar and log section"""
        row = start_row
        
        ttk.Label(parent, text="📋 Progress & Log:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        row += 1
        
        self.log_text = scrolledtext.ScrolledText(parent, height=8, width=80)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
    def browse_directory(self):
        """Browse for working directory"""
        directory = filedialog.askdirectory(initialdir=self.working_dir.get())
        if directory:
            self.working_dir.set(directory)
            self.update_file_counts()
            self.audio_controls.auto_find_audio(directory, self.log)
            
    def update_file_counts(self):
        """Update the file count display"""
        directory = self.working_dir.get()
        video_count, image_count = self.file_utils.count_files(directory)
        
        self.file_counts_label.config(
            text=f"📹 {video_count} videos, 📸 {image_count} images found"
        )
        
    def save_settings(self):
        """Save current settings to JSON file"""
        settings = self._collect_all_settings()
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            if self.settings_manager.save_settings(settings, filename):
                self.log(f"💾 Settings saved to {os.path.basename(filename)}")
            else:
                messagebox.showerror("Error", "Failed to save settings")
                
    def load_settings(self):
        """Load settings from JSON file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            settings = self.settings_manager.load_settings(filename)
            if settings:
                self._apply_all_settings(settings)
                self.log(f"📁 Settings loaded from {os.path.basename(filename)}")
                self.update_file_counts()
            else:
                messagebox.showerror("Error", "Failed to load settings")
                
    def _collect_all_settings(self):
        """Collect settings from all components"""
        settings = {
            'working_dir': self.working_dir.get(),
            'output_name': self.output_name.get()
        }
        
        # Add audio settings
        settings.update(self.audio_controls.get_settings())
        
        # Add parameter settings
        settings.update(self.parameter_tabs.get_settings())
        
        return settings
        
    def _apply_all_settings(self, settings):
        """Apply settings to all components"""
        # Apply main settings
        if 'working_dir' in settings:
            self.working_dir.set(settings['working_dir'])
        if 'output_name' in settings:
            self.output_name.set(settings['output_name'])
            
        # Apply audio settings
        self.audio_controls.apply_settings(settings)
        
        # Apply parameter settings
        self.parameter_tabs.apply_settings(settings)
        
    def create_montage(self):
        """Create the video montage"""
        # Validation
        if not os.path.exists(self.working_dir.get()):
            messagebox.showerror("Error", "Working directory does not exist")
            return
            
        if not self.audio_controls.validate_audio():
            messagebox.showerror("Error", "Please select a valid audio file")
            return
            
        # Start processing in separate thread
        self.progress.start()
        threading.Thread(target=self._create_montage_thread, daemon=True).start()
        
    def _create_montage_thread(self):
        """Thread function for creating montage"""
        try:
            self.log("🎬 Starting montage creation...")
            
            # Collect all settings
            settings = self._collect_all_settings()
            
            # Process the montage
            success = self.montage_processor.create_montage(
                working_dir=self.working_dir.get(),
                settings=settings,
                log_callback=self.log
            )
            
            if success:
                output_file = f"{self.output_name.get()}.mp4"
                self.log(f"🎉 SUCCESS! Created: {output_file}")
                messagebox.showinfo("Success", f"Montage created successfully!\nOutput: {output_file}")
            else:
                self.log("❌ Failed to create montage")
                messagebox.showerror("Error", "Failed to create montage. Check log for details.")
                
        except Exception as e:
            self.log(f"❌ Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.progress.stop()
            
    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()