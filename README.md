Alpha video tools.

-- Clean.py
IA based scene detector 
adjust scene threshold to 0.9 for best results
	
-- SequenceDector.py (ignore, was a an intermediate project)

-- videoSelector.py
Tool to select videos from a large quantity of videos, preview as loops, adjustable video speed.
Copies video into a subdir "selected"

-- autolooper.py
Optical flow based, Makes vjloops (detect sequence + blendin and an out)
check parameters to enhance experience. 

    parser.add_argument('--output', '-o', default='output_loop.mp4', help='Output video file path')
    parser.add_argument('--chunk-size', '-c', type=int, default=300, help='Number of frames to process at once')
    parser.add_argument('--min-loop-length', '-m', type=int, default=30, help='Minimum loop length in frames')
    parser.add_argument('--threshold', '-t', type=float, default=10.0, help='Similarity threshold (lower = more similar)')
    parser.add_argument('--blend-frames', '-b', type=int, default=15, help='Number of frames to blend for transitions')
    parser.add_argument('--candidates', '-n', type=int, default=3, help='Maximum number of loop candidates to generate per chunk')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing even if GPU is available')
    parser.add_argument('--benchmark', action='store_true', help='Run CPU vs GPU benchmark on this video')


-- automaticFfmpeginstaller_and_path.bat
Windows bat file to install ffmpeg and insert into path	

-- batchRenameMoviesincremental.bat
Windows bat file to rename files from 001.mp4 to ... xxxx.mp4

-- convertVideosforCablesGl.bat
Windows bat file for ffmpeg conversion for cables GL
