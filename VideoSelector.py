
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import shutil

class VideoGridPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("MP4 Video Grid Player")
        self.root.geometry("1000x800")
        
        # Variables
        self.videos = []
        self.current_directory = ""
        self.preview_threads = {}
        self.playing_videos = {}
        self.loop_mode = {}
        self.current_video = None
        self.playback_speed = 1.0
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Top frame for directory selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Video Directory:").pack(side=tk.LEFT)
        self.dir_entry = ttk.Entry(top_frame, width=50)
        self.dir_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(top_frame, text="Browse", command=self.browse_directory).pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Load Videos", command=self.load_videos).pack(side=tk.LEFT, padx=5)
        
        # Playback speed control
        speed_frame = ttk.Frame(self.root, padding="5")
        speed_frame.pack(fill=tk.X, padx=10)
        
        ttk.Label(speed_frame, text="Playback Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=1.0, to=4.0, orient=tk.HORIZONTAL, 
                                variable=self.speed_var, length=200,
                                command=self.update_speed)
        speed_scale.pack(side=tk.LEFT, padx=5)
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        # Main video display frame
        self.main_display_frame = ttk.Frame(self.root, padding="10")
        self.main_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Main video canvas
        self.main_canvas = tk.Canvas(self.main_display_frame, width=640, height=360, bg="black")
        self.main_canvas.pack(pady=10)
        
        # Main video controls
        main_controls = ttk.Frame(self.main_display_frame)
        main_controls.pack(pady=5)
        
        self.video_name_label = ttk.Label(main_controls, text="No video selected")
        self.video_name_label.pack(side=tk.TOP, pady=5)
        
        # Create a frame for the video grid with scrollbar
        self.grid_frame = ttk.Frame(self.root)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a canvas for scrolling
        self.canvas = tk.Canvas(self.grid_frame)
        scrollbar = ttk.Scrollbar(self.grid_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=10, pady=5)
        
    def update_speed(self, value):
        self.playback_speed = float(value)
        self.speed_label.config(text=f"{self.playback_speed:.1f}x")
        
    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)
            
    def load_videos(self):
        directory = self.dir_entry.get()
        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Error", "Please select a valid directory")
            return
            
        self.current_directory = directory
        self.videos = []
        
        # Stop any playing videos
        for video_id in list(self.playing_videos.keys()):
            self.stop_preview(video_id)
        
        # Reset main display
        self.current_video = None
        self.video_name_label.config(text="No video selected")
        self.main_canvas.delete("all")
        
        # Clear grid
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Find all MP4 files
        for file in os.listdir(directory):
            if file.lower().endswith(".mp4"):
                self.videos.append(file)
        
        if not self.videos:
            self.status_var.set("No MP4 videos found in the selected directory")
            return
        
        # Create selected directory if it doesn't exist
        selected_dir = os.path.join(self.current_directory, "selected")
        if not os.path.exists(selected_dir):
            os.makedirs(selected_dir)
        
        self.status_var.set(f"Found {len(self.videos)} MP4 videos")
        self.display_video_grid()
        
    def display_video_grid(self):
        # Configure grid
        cols = 3  # Number of columns in the grid
        
        # Create frames for each video
        for i, video in enumerate(self.videos):
            row = i // cols
            col = i % cols
            
            # Create frame for this video
            video_frame = ttk.Frame(self.scrollable_frame, padding=5)
            video_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Video ID
            video_id = f"video_{i}"
            
            # Create thumbnail canvas
            canvas = tk.Canvas(video_frame, width=200, height=150, bg="black")
            canvas.pack(pady=5)
            
            # Make thumbnail clickable to show in main display
            canvas.bind("<Button-1>", lambda e, vid=video_id, file=video: 
                        self.show_in_main_display(vid, file))
            
            # Load first frame as thumbnail
            threading.Thread(target=self.load_thumbnail, 
                             args=(os.path.join(self.current_directory, video), canvas, video_id)).start()
            
            # Video name label
            ttk.Label(video_frame, text=os.path.basename(video), wraplength=200).pack(pady=2)
            
            # Control buttons frame
            control_frame = ttk.Frame(video_frame)
            control_frame.pack(fill=tk.X, pady=5)
            
            # Play button
            play_btn = ttk.Button(control_frame, text="Play", 
                                  command=lambda vid=video_id, file=video: self.play_preview(vid, file))
            play_btn.pack(side=tk.LEFT, padx=2)
            
            # Stop button
            stop_btn = ttk.Button(control_frame, text="Stop", 
                                  command=lambda vid=video_id: self.stop_preview(vid))
            stop_btn.pack(side=tk.LEFT, padx=2)
            
            # Loop checkbox (default to checked)
            self.loop_mode[video_id] = tk.BooleanVar(value=True)
            loop_chk = ttk.Checkbutton(control_frame, text="Loop", variable=self.loop_mode[video_id])
            loop_chk.pack(side=tk.LEFT, padx=2)
            
            # Select button
            select_btn = ttk.Button(control_frame, text="Select", 
                                  command=lambda file=video: self.select_video(file))
            select_btn.pack(side=tk.LEFT, padx=2)
            
            # Store canvas reference
            self.playing_videos[video_id] = {"canvas": canvas, "playing": False}
    
    def load_thumbnail(self, video_path, canvas, video_id):
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                # Convert to a thumbnail
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (200, 150))
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update canvas - needs to be done in main thread
                self.root.after(0, lambda: self.update_canvas(canvas, img_tk, video_id))
            cap.release()
        except Exception as e:
            print(f"Error loading thumbnail: {e}")
    
    def update_canvas(self, canvas, img_tk, video_id):
        canvas.delete("all")
        canvas.image = img_tk  # Keep a reference
        canvas.create_image(100, 75, image=img_tk)
        self.playing_videos[video_id]["thumbnail"] = img_tk
    
    def show_in_main_display(self, video_id, video):
        # Update current video
        self.current_video = video
        self.video_name_label.config(text=video)
        
        # Stop any playing video in main display
        if hasattr(self, 'main_video_thread') and self.main_video_thread is not None:
            self.main_video_playing = False
            self.main_video_thread.join(0.5)
        
        # Start playing in main display
        self.main_video_playing = True
        video_path = os.path.join(self.current_directory, video)
        
        self.main_video_thread = threading.Thread(target=self.play_in_main_display, 
                                                 args=(video_path,))
        self.main_video_thread.daemon = True
        self.main_video_thread.start()
    
    def play_in_main_display(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Default FPS if not available
            
            while self.main_video_playing:
                ret, frame = cap.read()
                
                if not ret:
                    # Loop back to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Convert to a format Tkinter can display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 360))
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update main canvas - needs to be done in main thread
                self.root.after(0, lambda: self.update_main_canvas(img_tk))
                
                # Control frame rate (adjust by playback speed)
                time.sleep(1 / (fps * self.playback_speed))
            
            cap.release()
        except Exception as e:
            print(f"Error playing video in main display: {e}")
            self.main_video_playing = False
    
    def update_main_canvas(self, img_tk):
        self.main_canvas.delete("all")
        self.main_canvas.image = img_tk  # Keep a reference
        self.main_canvas.create_image(320, 180, image=img_tk)
    
    def play_preview(self, video_id, video):
        # Stop if already playing
        if video_id in self.playing_videos and self.playing_videos[video_id]["playing"]:
            self.stop_preview(video_id)
        
        # Start playing
        self.playing_videos[video_id]["playing"] = True
        video_path = os.path.join(self.current_directory, video)
        canvas = self.playing_videos[video_id]["canvas"]
        
        # Show in main display too
        self.show_in_main_display(video_id, video)
        
        # Create a new thread for playing
        thread = threading.Thread(target=self.play_video, args=(video_path, canvas, video_id))
        thread.daemon = True
        thread.start()
        
        self.preview_threads[video_id] = thread
    
    def play_video(self, video_path, canvas, video_id):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Default FPS if not available
            
            while self.playing_videos[video_id]["playing"]:
                ret, frame = cap.read()
                
                if not ret:
                    # If video ended and loop is enabled, restart
                    if self.loop_mode[video_id].get():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Convert to a format Tkinter can display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (200, 150))
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update canvas - needs to be done in main thread
                self.root.after(0, lambda: self.update_canvas(canvas, img_tk, video_id))
                
                # Control frame rate (adjust by playback speed)
                time.sleep(1 / (fps * self.playback_speed))
            
            cap.release()
            
            # Reset to thumbnail if not playing
            if video_id in self.playing_videos and not self.playing_videos[video_id]["playing"]:
                self.root.after(0, lambda: self.display_thumbnail(video_id))
                
        except Exception as e:
            print(f"Error playing video: {e}")
            self.playing_videos[video_id]["playing"] = False
    
    def display_thumbnail(self, video_id):
        if "thumbnail" in self.playing_videos[video_id]:
            canvas = self.playing_videos[video_id]["canvas"]
            img_tk = self.playing_videos[video_id]["thumbnail"]
            canvas.delete("all")
            canvas.create_image(100, 75, image=img_tk)
    
    def stop_preview(self, video_id):
        if video_id in self.playing_videos:
            self.playing_videos[video_id]["playing"] = False
            
            # Wait for thread to finish
            if video_id in self.preview_threads:
                self.preview_threads[video_id].join(0.5)
                del self.preview_threads[video_id]
    
    def select_video(self, video):
        if messagebox.askyesno("Confirm Selection", f"Copy {video} to the 'selected' folder?"):
            try:
                # Create selected directory if it doesn't exist
                selected_dir = os.path.join(self.current_directory, "selected")
                if not os.path.exists(selected_dir):
                    os.makedirs(selected_dir)
                
                # Copy the file instead of moving it
                src_path = os.path.join(self.current_directory, video)
                dest_path = os.path.join(selected_dir, video)
                
                # If the file already exists in the selected folder, ask for confirmation to overwrite
                if os.path.exists(dest_path):
                    if not messagebox.askyesno("File Exists", 
                                             f"{video} already exists in the selected folder. Overwrite?"):
                        return
                
                shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
                
                messagebox.showinfo("Success", f"{video} has been copied to the selected folder")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not copy the video: {e}")

def main():
    root = tk.Tk()
    app = VideoGridPlayer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
