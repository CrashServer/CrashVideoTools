"""
Montage Processor
Main logic for creating video montages
"""

import os
import subprocess
from .script_generator import ScriptGenerator

class MontageProcessor:
    def __init__(self):
        self.script_generator = ScriptGenerator()
        
    def create_montage(self, working_dir, settings, log_callback):
        """Create video montage with given settings"""
        try:
            log_callback("🎬 Starting montage creation...")
            
            # Save current directory
            original_dir = os.getcwd()
            
            # Change to working directory
            os.chdir(working_dir)
            
            # Create clips directory
            os.makedirs("clips", exist_ok=True)
            
            # Generate bash script
            script_content = self.script_generator.generate_script(settings)
            
            # Write script to temp file
            script_path = "temp_montage_script.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
                
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Run script
            log_callback("🔄 Running montage script...")
            process = subprocess.Popen(
                ["bash", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to log
            for line in process.stdout:
                log_callback(line.strip())
                
            process.wait()
            
            # Check result
            output_file = f"{settings.get('output_name', 'MONTAGE')}.mp4"
            success = os.path.exists(output_file)
            
            # Cleanup
            if os.path.exists(script_path):
                os.remove(script_path)
                
            return success
            
        except Exception as e:
            log_callback(f"❌ Error in montage processor: {e}")
            return False
        finally:
            # Always return to original directory
            os.chdir(original_dir)