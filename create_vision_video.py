"""
Create Vision Video

Simple script to create videos from vision analysis images.
"""

import sys
import os

# Add sim package to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim'))

from sim.utils.video_creator import main

if __name__ == "__main__":
    main()