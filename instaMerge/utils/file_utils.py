"""
File Utilities
Helper functions for file operations
"""

import os
import glob

class FileUtils:
    @staticmethod
    def count_files(directory):
        """Count video and image files in directory"""
        if not os.path.exists(directory):
            return 0, 0
            
        # Count video files
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV"]
        video_count = 0
        for ext in video_extensions:
            video_count += len(glob.glob(os.path.join(directory, ext)))
            
        # Count image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"]
        image_count = 0
        for ext in image_extensions:
            image_count += len(glob.glob(os.path.join(directory, ext)))
            
        return video_count, image_count
        
    @staticmethod
    def get_video_files(directory):
        """Get list of video files in directory"""
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV"]
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(directory, ext)))
            
        return [os.path.basename(f) for f in video_files]
        
    @staticmethod
    def get_image_files(directory):
        """Get list of image files in directory"""
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"]
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
            
        return [os.path.basename(f) for f in image_files]
        
    @staticmethod
    def get_audio_files(directory):
        """Get list of audio/video files that can be used as audio source"""
        audio_extensions = ["*.mp4", "*.mov", "*.mkv", "*.avi", "*.wav", "*.mp3", "*.aac", "*.flac", "*.ogg"]
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(directory, ext)))
            
        return [os.path.basename(f) for f in audio_files]
        
    @staticmethod
    def is_valid_audio_format(filename):
        """Check if file is a valid audio format"""
        valid_extensions = ['.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.aac', '.flac', '.ogg']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)
        
    @staticmethod
    def is_valid_video_format(filename):
        """Check if file is a valid video format"""
        valid_extensions = ['.mp4', '.mov', '.mkv', '.avi']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)
        
    @staticmethod
    def is_valid_image_format(filename):
        """Check if file is a valid image format"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)