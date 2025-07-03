"""
Settings Manager
Handles saving and loading of application settings
"""

import json
import os

class SettingsManager:
    @staticmethod
    def save_settings(settings, filename):
        """Save settings to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
            
    @staticmethod
    def load_settings(filename):
        """Load settings from JSON file"""
        try:
            if not os.path.exists(filename):
                return None
                
            with open(filename, 'r') as f:
                settings = json.load(f)
            return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            return None
            
    @staticmethod
    def get_default_settings():
        """Get default application settings"""
        return {
            'working_dir': os.getcwd(),
            'audio_file': '',
            'output_name': 'MONTAGE',
            'audio_start_time': 0.0,
            'audio_end_time': 0.0,
            'use_audio_timing': False,
            'max_videos': 25,
            'max_images': 10,
            'video_duration': 1.8,
            'image_duration': 0.3,
            'speed_multiplier': 2.0,
            'video_quality': 23,
            'image_quality': 18,
            'contrast': 1.2,
            'saturation': 1.3,
            'brightness': 0.0,
            'match_audio_length': True,
            'random_shuffle': True,
            'glitch_effects': True,
            'image_flash_intensity': 3.0
        }
        
    @staticmethod
    def validate_settings(settings):
        """Validate settings dictionary"""
        defaults = SettingsManager.get_default_settings()
        
        # Check if all required keys exist
        for key in defaults:
            if key not in settings:
                settings[key] = defaults[key]
                
        # Validate value ranges
        settings['max_videos'] = max(1, min(50, settings['max_videos']))
        settings['max_images'] = max(1, min(20, settings['max_images']))
        settings['video_duration'] = max(0.1, min(10.0, settings['video_duration']))
        settings['image_duration'] = max(0.1, min(2.0, settings['image_duration']))
        settings['speed_multiplier'] = max(0.5, min(5.0, settings['speed_multiplier']))
        settings['video_quality'] = max(15, min(35, settings['video_quality']))
        settings['image_quality'] = max(15, min(35, settings['image_quality']))
        settings['contrast'] = max(0.1, min(5.0, settings['contrast']))
        settings['saturation'] = max(0.1, min(5.0, settings['saturation']))
        settings['brightness'] = max(-1.0, min(1.0, settings['brightness']))
        settings['image_flash_intensity'] = max(1.0, min(5.0, settings['image_flash_intensity']))
        
        return settings