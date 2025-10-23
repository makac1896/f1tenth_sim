#!/usr/bin/env python3
"""
F1Tenth Simulator GUI Launcher

A GUI menu system for running all F1Tenth simulation scripts with parameters.
Outputs results to terminal while providing an easy-to-use interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import os
import sys
import glob
from datetime import datetime


class F1TenthSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("F1Tenth Simulator - Script Launcher")
        self.root.geometry("1400x800")
        
        # Current working directory
        self.cwd = os.getcwd()
        
        # Find Python executable (prefer virtual environment)
        self.python_exe = self.find_python_executable()
        
        # Create main layout
        self.create_widgets()
        
        # Initialize results (after a short delay to ensure GUI is ready)
        self.root.after(500, self.initialize_results)
        
        # Terminal output
        self.terminal_output = []
    
    def find_python_executable(self):
        """Find the correct Python executable (prefer virtual environment)"""
        
        # Check for virtual environment Python first
        venv_paths = [
            os.path.join(self.cwd, "f1tenth_sim_env", "Scripts", "python.exe"),  # Windows
            os.path.join(self.cwd, "f1tenth_sim_env", "bin", "python"),         # Linux/Mac
            os.path.join(self.cwd, "venv", "Scripts", "python.exe"),            # Alt Windows
            os.path.join(self.cwd, "venv", "bin", "python"),                    # Alt Linux/Mac
        ]
        
        for venv_python in venv_paths:
            if os.path.exists(venv_python):
                self.log_to_terminal(f"[OK] Found virtual environment Python: {venv_python}")
                return venv_python
        
        # Fallback to system Python
        self.log_to_terminal(f"⚠️ No virtual environment found, using system Python: {sys.executable}")
        return sys.executable
        
    def create_widgets(self):
        """Create the main GUI layout with split screen (left/right)"""
        
        # Create main paned window for horizontal split layout
        main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left frame for script controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Notebook for different script categories in left frame
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill='both', expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        notebook = self.notebook
        
        # Analysis Scripts Tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        self.create_analysis_tab(analysis_frame)
        
        # Vision Scripts Tab
        vision_frame = ttk.Frame(notebook)
        notebook.add(vision_frame, text="Vision")
        self.create_vision_tab(vision_frame)
        
        # Test Scripts Tab
        test_frame = ttk.Frame(notebook)
        notebook.add(test_frame, text="Testing")
        self.create_test_tab(test_frame)
        
        # Utility Scripts Tab
        utility_frame = ttk.Frame(notebook)
        notebook.add(utility_frame, text="Utilities")
        self.create_utility_tab(utility_frame)
        
        # Results Viewer Tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        self.create_results_tab(results_frame)
        
        # Right frame for terminal output (always visible)
        right_frame = ttk.Frame(main_paned)
        right_frame.configure(width=600)  # Minimum width for terminal
        main_paned.add(right_frame, weight=1)
        self.create_terminal_section(right_frame)

    def create_analysis_tab(self, parent):
        """Analysis scripts interface"""
        
        # Gap Analysis Section
        gap_frame = ttk.LabelFrame(parent, text="Gap Analysis", padding=10)
        gap_frame.pack(fill='x', padx=10, pady=5)
        
        # Parameters frame
        params_frame = ttk.Frame(gap_frame)
        params_frame.pack(fill='x', pady=5)
        
        # Frames parameter
        ttk.Label(params_frame, text="Frames:").grid(row=0, column=0, sticky='w', padx=5)
        self.gap_frames = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.gap_frames, width=10).grid(row=0, column=1, padx=5)
        
        # Start parameter
        ttk.Label(params_frame, text="Start:").grid(row=0, column=2, sticky='w', padx=5)
        self.gap_start = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.gap_start, width=10).grid(row=0, column=3, padx=5)
        
        # Lidar directory
        ttk.Label(params_frame, text="Lidar Dir:").grid(row=1, column=0, sticky='w', padx=5)
        self.gap_lidar_dir = tk.StringVar(value="all_logs/lidar/lidar_1")
        ttk.Entry(params_frame, textvariable=self.gap_lidar_dir, width=30).grid(row=1, column=1, columnspan=2, padx=5, sticky='ew')
        
        # Vision directory
        ttk.Label(params_frame, text="Vision Dir:").grid(row=2, column=0, sticky='w', padx=5)
        self.gap_vision_dir = tk.StringVar(value="all_logs/vision/vision_1")
        ttk.Entry(params_frame, textvariable=self.gap_vision_dir, width=30).grid(row=2, column=1, columnspan=2, padx=5, sticky='ew')
        
        # Options
        options_frame = ttk.Frame(gap_frame)
        options_frame.pack(fill='x', pady=5)
        
        self.gap_verbose = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Verbose", variable=self.gap_verbose).pack(side='left', padx=5)
        
        self.gap_visualize = tk.BooleanVar(value=True)  # Default to True
        ttk.Checkbutton(options_frame, text="Visualize", variable=self.gap_visualize).pack(side='left', padx=5)
        
        # Run button
        ttk.Button(gap_frame, text="Run Gap Analysis", 
                  command=self.run_gap_analysis).pack(pady=10)
        
        # Batch Analysis Section
        batch_frame = ttk.LabelFrame(parent, text="Batch Analysis", padding=10)
        batch_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(batch_frame, text="Run multiple pre-configured analysis scenarios").pack()
        ttk.Button(batch_frame, text="Run Batch Analysis", 
                  command=self.run_batch_analysis).pack(pady=10)

    def create_vision_tab(self, parent):
        """Vision scripts interface"""
        
        # Vision Gap Analysis
        vision_gap_frame = ttk.LabelFrame(parent, text="Vision Gap Analysis", padding=10)
        vision_gap_frame.pack(fill='x', padx=10, pady=5)
        
        params_frame = ttk.Frame(vision_gap_frame)
        params_frame.pack(fill='x', pady=5)
        
        # Parameters
        ttk.Label(params_frame, text="Vision Dir:").grid(row=0, column=0, sticky='w', padx=5)
        self.vision_gap_dir = tk.StringVar(value="all_logs/vision/vision_1")
        ttk.Entry(params_frame, textvariable=self.vision_gap_dir, width=30).grid(row=0, column=1, columnspan=2, padx=5, sticky='ew')
        
        ttk.Label(params_frame, text="Frames:").grid(row=1, column=0, sticky='w', padx=5)
        self.vision_gap_frames = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.vision_gap_frames, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(params_frame, text="Start:").grid(row=1, column=2, sticky='w', padx=5)
        self.vision_gap_start = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.vision_gap_start, width=10).grid(row=1, column=3, padx=5)
        
        # Options
        self.vision_gap_visualize = tk.BooleanVar(value=True)  # Default to True
        ttk.Checkbutton(vision_gap_frame, text="Save visualization images", 
                       variable=self.vision_gap_visualize).pack(pady=5)
        
        ttk.Button(vision_gap_frame, text="Run Vision Gap Analysis", 
                  command=self.run_vision_gap_analysis).pack(pady=10)
        
        # Vision Safety Analysis
        safety_frame = ttk.LabelFrame(parent, text="Vision Safety Analysis", padding=10)
        safety_frame.pack(fill='x', padx=10, pady=5)
        
        safety_params = ttk.Frame(safety_frame)
        safety_params.pack(fill='x', pady=5)
        
        ttk.Label(safety_params, text="Vision Dir:").grid(row=0, column=0, sticky='w', padx=5)
        self.safety_vision_dir = tk.StringVar(value="all_logs/vision/vision_1")
        ttk.Entry(safety_params, textvariable=self.safety_vision_dir, width=30).grid(row=0, column=1, columnspan=2, padx=5, sticky='ew')
        
        ttk.Label(safety_params, text="Frames:").grid(row=1, column=0, sticky='w', padx=5)
        self.safety_frames = tk.StringVar(value="10")
        ttk.Entry(safety_params, textvariable=self.safety_frames, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(safety_params, text="Safety Distance:").grid(row=1, column=2, sticky='w', padx=5)
        self.safety_distance = tk.StringVar(value="1.0")
        ttk.Entry(safety_params, textvariable=self.safety_distance, width=10).grid(row=1, column=3, padx=5)
        
        ttk.Button(safety_frame, text="Run Vision Safety Analysis", 
                  command=self.run_vision_safety_analysis).pack(pady=10)
        
        # Vision Processing Stages
        stages_frame = ttk.LabelFrame(parent, text="Vision Processing Stages", padding=10)
        stages_frame.pack(fill='x', padx=10, pady=5)
        
        stages_params = ttk.Frame(stages_frame)
        stages_params.pack(fill='x', pady=5)
        
        ttk.Label(stages_params, text="Vision Dir:").grid(row=0, column=0, sticky='w', padx=5)
        self.stages_vision_dir = tk.StringVar(value="all_logs/vision/vision_1")
        ttk.Entry(stages_params, textvariable=self.stages_vision_dir, width=30).grid(row=0, column=1, columnspan=2, padx=5, sticky='ew')
        
        ttk.Label(stages_params, text="Frames:").grid(row=1, column=0, sticky='w', padx=5)
        self.stages_frames = tk.StringVar(value="3")
        ttk.Entry(stages_params, textvariable=self.stages_frames, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(stages_params, text="Start:").grid(row=1, column=2, sticky='w', padx=5)
        self.stages_start = tk.StringVar(value="0")
        ttk.Entry(stages_params, textvariable=self.stages_start, width=10).grid(row=1, column=3, padx=5)
        
        ttk.Button(stages_frame, text="Run Vision Stages Test", 
                  command=self.run_vision_stages).pack(pady=10)
        
        # Basic Vision Analysis
        basic_frame = ttk.LabelFrame(parent, text="Basic Vision Analysis", padding=10)
        basic_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(basic_frame, text="Basic edge detection processing for vision frames").pack(pady=5)
        
        basic_params = ttk.Frame(basic_frame)
        basic_params.pack(fill='x', pady=5)
        basic_params.columnconfigure(1, weight=1)
        
        ttk.Label(basic_params, text="Vision Dir:").grid(row=0, column=0, sticky='w', padx=5)
        self.basic_vision_dir = tk.StringVar(value="all_logs/vision/vision_1")
        ttk.Entry(basic_params, textvariable=self.basic_vision_dir, width=30).grid(row=0, column=1, columnspan=2, padx=5, sticky='ew')
        
        ttk.Label(basic_params, text="Frames:").grid(row=1, column=0, sticky='w', padx=5)
        self.basic_frames = tk.StringVar(value="10")
        ttk.Entry(basic_params, textvariable=self.basic_frames, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(basic_params, text="Start:").grid(row=1, column=2, sticky='w', padx=5)
        self.basic_start = tk.StringVar(value="0")
        ttk.Entry(basic_params, textvariable=self.basic_start, width=10).grid(row=1, column=3, padx=5)
        
        ttk.Button(basic_frame, text="Run Basic Vision Analysis", 
                  command=self.run_basic_vision_analysis).pack(pady=10)

    def create_test_tab(self, parent):
        """Test scripts interface"""
        
        # Simple Test
        simple_frame = ttk.LabelFrame(parent, text="Simple Tests", padding=10)
        simple_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(simple_frame, text="Test Simulator", 
                  command=self.run_test_simulator).pack(side='left', padx=10, pady=10)
        
        ttk.Button(simple_frame, text="Test Vision Algorithm", 
                  command=self.run_test_vision).pack(side='left', padx=10, pady=10)
        
        # Algorithm Tests
        algo_frame = ttk.LabelFrame(parent, text="Algorithm Tests", padding=10)
        algo_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(algo_frame, text="Quick tests of individual algorithms").pack(pady=5)
        
        buttons_frame = ttk.Frame(algo_frame)
        buttons_frame.pack(pady=10)
        
        ttk.Button(buttons_frame, text="Test Vision AEB", 
                  command=self.run_test_vision_aeb).pack(side='left', padx=10)
        
        ttk.Button(buttons_frame, text="Test Lidar Gap Follow", 
                  command=self.run_test_lidar_gap).pack(side='left', padx=10)

    def create_utility_tab(self, parent):
        """Utility scripts interface"""
        
        # Timestamp Analysis
        timestamp_frame = ttk.LabelFrame(parent, text="Data Analysis", padding=10)
        timestamp_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(timestamp_frame, text="Analyze timestamp alignment between lidar and vision data").pack(pady=5)
        ttk.Button(timestamp_frame, text="Analyze Timestamps", 
                  command=self.run_timestamp_analysis).pack(pady=10)
        
        # File Conversion
        convert_frame = ttk.LabelFrame(parent, text="File Conversion", padding=10)
        convert_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(convert_frame, text="Convert raw image files to PNG format").pack(pady=5)
        
        convert_params = ttk.Frame(convert_frame)
        convert_params.pack(fill='x', pady=5)
        
        ttk.Label(convert_params, text="Input File:").grid(row=0, column=0, sticky='w', padx=5)
        self.convert_input = tk.StringVar()
        ttk.Entry(convert_params, textvariable=self.convert_input, width=40).grid(row=0, column=1, padx=5, sticky='ew')
        ttk.Button(convert_params, text="Browse", 
                  command=self.browse_convert_file).grid(row=0, column=2, padx=5)
        
        ttk.Button(convert_frame, text="Convert to PNG", 
                  command=self.run_convert_file).pack(pady=10)
        
        # Video Creation
        video_frame = ttk.LabelFrame(parent, text="Video Generation", padding=10)
        video_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(video_frame, text="Create videos from vision analysis images").pack(pady=5)
        
        video_params = ttk.Frame(video_frame)
        video_params.pack(fill='x', pady=5)
        video_params.columnconfigure(1, weight=1)
        
        ttk.Label(video_params, text="Input Dir:").grid(row=0, column=0, sticky='w', padx=5)
        self.video_input_dir = tk.StringVar(value="images/vision")
        ttk.Entry(video_params, textvariable=self.video_input_dir, width=30).grid(row=0, column=1, padx=5, sticky='ew')
        
        ttk.Label(video_params, text="Output:").grid(row=1, column=0, sticky='w', padx=5)
        self.video_output = tk.StringVar(value="vision_analysis.mp4")
        ttk.Entry(video_params, textvariable=self.video_output, width=25).grid(row=1, column=1, padx=5, sticky='w')
        
        ttk.Label(video_params, text="FPS:").grid(row=1, column=2, sticky='w', padx=5)
        self.video_fps = tk.StringVar(value="10")
        ttk.Entry(video_params, textvariable=self.video_fps, width=8).grid(row=1, column=3, padx=5)
        
        ttk.Button(video_frame, text="Create Video", 
                  command=self.run_create_video).pack(pady=10)
        
        # Depth Demo
        depth_frame = ttk.LabelFrame(parent, text="Demonstrations", padding=10)
        depth_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(depth_frame, text="Interactive demonstrations and examples").pack(pady=5)
        ttk.Button(depth_frame, text="Depth Gap Selection Demo", 
                  command=self.run_depth_demo).pack(pady=10)

    def create_results_tab(self, parent):
        """Results viewer interface"""
        
        # Image Browser Section
        image_frame = ttk.LabelFrame(parent, text="Generated Images", padding=10)
        image_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Directory selection
        dir_frame = ttk.Frame(image_frame)
        dir_frame.pack(fill='x', pady=5)
        
        ttk.Label(dir_frame, text="Image Directory:").pack(side='left', padx=5)
        self.results_dir = tk.StringVar(value="images/vision")
        results_combo = ttk.Combobox(dir_frame, textvariable=self.results_dir, width=30)
        results_combo['values'] = ('images/vision', 'images/vision/stages', 'images/vision/aeb', 'images/lidar')
        results_combo.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(dir_frame, text="Refresh", command=self.refresh_results).pack(side='right', padx=5)
        ttk.Button(dir_frame, text="Open Folder", command=self.open_results_folder).pack(side='right', padx=5)
        
        # Image list and preview
        preview_frame = ttk.Frame(image_frame)
        preview_frame.pack(fill='both', expand=True, pady=5)
        
        # Left side - file list
        list_frame = ttk.Frame(preview_frame)
        list_frame.pack(side='left', fill='y', padx=(0, 5))
        
        ttk.Label(list_frame, text="Files:").pack(anchor='w')
        self.results_listbox = tk.Listbox(list_frame, width=30, height=15)
        self.results_listbox.pack(fill='both', expand=True)
        self.results_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.results_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.results_listbox.config(yscrollcommand=scrollbar.set)
        
        # Right side - preview area
        self.preview_frame = ttk.Frame(preview_frame)
        self.preview_frame.pack(side='right', fill='both', expand=True)
        
        ttk.Label(self.preview_frame, text="Preview").pack(pady=5)
        self.preview_label = ttk.Label(self.preview_frame, text="Select an image to preview")
        self.preview_label.pack(expand=True)
        
        # Video Section
        video_frame = ttk.LabelFrame(parent, text="Generated Videos", padding=10)
        video_frame.pack(fill='x', padx=10, pady=5)
        
        video_controls = ttk.Frame(video_frame)
        video_controls.pack(fill='x', pady=5)
        
        ttk.Label(video_controls, text="Videos Directory:").pack(side='left', padx=5)
        self.video_dir = tk.StringVar(value="videos/vision")
        video_dir_entry = ttk.Entry(video_controls, textvariable=self.video_dir, width=30)
        video_dir_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(video_controls, text="Open Video Folder", command=self.open_video_folder).pack(side='right', padx=5)
        ttk.Button(video_controls, text="Create Video from Current Images", command=self.create_video_from_current).pack(side='right', padx=5)
        
        # Video list
        self.video_listbox = tk.Listbox(video_frame, height=5)
        self.video_listbox.pack(fill='x', pady=5)
        self.video_listbox.bind('<Double-1>', self.play_video)

    def create_terminal_section(self, parent):
        """Terminal output display in bottom pane"""
        
        # Terminal header with controls
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', padx=5, pady=(5, 0))
        
        ttk.Label(header_frame, text="Terminal Output", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        
        # Control buttons on the right
        ttk.Button(header_frame, text="Clear", 
                  command=self.clear_terminal).pack(side='right', padx=(5, 0))
        
        ttk.Button(header_frame, text="Save", 
                  command=self.save_terminal_output).pack(side='right', padx=(5, 0))
        
        # Terminal text area
        self.terminal_text = scrolledtext.ScrolledText(parent, wrap='word', height=12)
        self.terminal_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initial welcome message
        welcome_msg = f"""F1Tenth Simulator GUI - Ready
Working Directory: {self.cwd}
Python Environment: {self.python_exe}
{'=' * 60}
Select a script from the tabs above to begin analysis.
Terminal output will appear here in real-time.
"""
        self.log_to_terminal(welcome_msg)

    def run_command(self, cmd, description="Running command"):
        """Run a command in a separate thread and display output"""
        
        def run_thread():
            self.log_to_terminal(f"\n{'='*60}")
            self.log_to_terminal(f"{description}")
            self.log_to_terminal(f"Command: {' '.join(cmd)}")
            self.log_to_terminal(f"{'='*60}\n")
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.cwd,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Read output line by line
                for line in iter(process.stdout.readline, ''):
                    self.log_to_terminal(line.rstrip())
                
                process.wait()
                
                if process.returncode == 0:
                    self.log_to_terminal(f"\n[OK] {description} completed successfully")
                else:
                    self.log_to_terminal(f"\n[ERROR] {description} failed with exit code {process.returncode}")
                    
            except Exception as e:
                self.log_to_terminal(f"\n[ERROR] Error running {description}: {e}")
        
        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=run_thread)
        thread.daemon = True
        thread.start()

    def log_to_terminal(self, message):
        """Add message to terminal output (thread-safe)"""
        def update():
            self.terminal_text.insert('end', message + '\n')
            self.terminal_text.see('end')
            self.root.update_idletasks()
        
        self.root.after(0, update)

    # Analysis script runners
    def run_gap_analysis(self):
        cmd = [
            self.python_exe, "gap_analysis.py",
            "--frames", self.gap_frames.get(),
            "--start", self.gap_start.get(),
            "--lidar-dir", self.gap_lidar_dir.get(),
            "--vision-dir", self.gap_vision_dir.get()
        ]
        
        if self.gap_verbose.get():
            cmd.append("--verbose")
        if self.gap_visualize.get():
            cmd.append("--visualize")
            
        self.run_command(cmd, "Gap Analysis")

    def run_batch_analysis(self):
        cmd = [self.python_exe, "run_batch_analysis.py"]
        self.run_command(cmd, "Batch Analysis")

    # Vision script runners
    def run_vision_gap_analysis(self):
        cmd = [
            self.python_exe, "vision_gap_analysis.py",
            "--vision_dir", self.vision_gap_dir.get(),
            "--frames", self.vision_gap_frames.get(),
            "--start", self.vision_gap_start.get()
        ]
        
        if self.vision_gap_visualize.get():
            cmd.append("--visualize")
            
        self.run_command_with_video(cmd, "Vision Gap Analysis", "images/vision")

    def run_vision_safety_analysis(self):
        cmd = [
            self.python_exe, "vision_safety_analysis.py",
            "--vision-dir", self.safety_vision_dir.get(),
            "--frames", self.safety_frames.get(),
            "--safety-distance", self.safety_distance.get()
        ]
        
        self.run_command(cmd, "Vision Safety Analysis")

    def run_vision_stages(self):
        cmd = [
            self.python_exe, "test_vision_stages.py",
            "--vision_dir", self.stages_vision_dir.get(),
            "--frames", self.stages_frames.get(),
            "--start", self.stages_start.get()
        ]
        
        self.run_command_with_video(cmd, "Vision Processing Stages", "images/vision/stages")

    # Test script runners
    def run_test_simulator(self):
        cmd = [self.python_exe, "test_simulator.py"]
        self.run_command(cmd, "Simulator Test")

    def run_test_vision(self):
        cmd = [self.python_exe, "test_vision.py"]
        self.run_command(cmd, "Vision Algorithm Test")

    def run_test_vision_aeb(self):
        cmd = [self.python_exe, "-c", "from sim.algorithms.vision_aeb_safety import test_vision_aeb; test_vision_aeb()"]
        self.run_command(cmd, "Vision AEB Test")

    def run_test_lidar_gap(self):
        cmd = [self.python_exe, "-c", "from sim.algorithms.lidar_gap_follow import LidarGapFollower; print('Lidar Gap Follow Test - TODO: Add test function')"]
        self.run_command(cmd, "Lidar Gap Follow Test")

    # Utility script runners
    def run_timestamp_analysis(self):
        cmd = [self.python_exe, "analyze_timestamps.py"]
        self.run_command(cmd, "Timestamp Analysis")

    def browse_convert_file(self):
        filename = filedialog.askopenfilename(
            title="Select RAW image file",
            filetypes=[("RAW files", "*.raw"), ("All files", "*.*")]
        )
        if filename:
            self.convert_input.set(filename)

    def run_convert_file(self):
        if not self.convert_input.get():
            messagebox.showerror("Error", "Please select an input file")
            return
            
        cmd = [
            self.python_exe, "-c", 
            f"from raw_to_png_converter import raw_to_png; print(raw_to_png('{self.convert_input.get()}', 'converted'))"
        ]
        
        self.run_command(cmd, "File Conversion")

    def run_create_video(self):
        if not self.video_input_dir.get():
            messagebox.showerror("Error", "Please specify input directory")
            return
            
        cmd = [
            self.python_exe, "create_vision_video.py",
            "--input-dir", self.video_input_dir.get(),
            "--output", self.video_output.get(),
            "--fps", self.video_fps.get()
        ]
        
        self.run_command(cmd, "Video Creation")

    def run_depth_demo(self):
        cmd = [self.python_exe, "depth_gap_selection_demo.py"]
        self.run_command(cmd, "Depth Gap Selection Demo")

    def run_basic_vision_analysis(self):
        cmd = [
            self.python_exe, "vision_analysis.py",
            "--vision-dir", self.basic_vision_dir.get(),
            "--frames", self.basic_frames.get(),
            "--start", self.basic_start.get()
        ]
        
        self.run_command_with_video(cmd, "Basic Vision Analysis", "images/vision")

    # Results viewer methods
    def refresh_results(self):
        """Refresh the results file list"""
        results_dir = self.results_dir.get()
        self.results_listbox.delete(0, tk.END)
        
        if os.path.exists(results_dir):
            try:
                files = []
                for file in os.listdir(results_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        files.append(file)
                
                # Sort files naturally (handle numbers properly)
                files.sort()
                
                for file in files:
                    self.results_listbox.insert(tk.END, file)
                    
                # Also refresh video list
                self.refresh_videos()
                
            except Exception as e:
                self.log_to_terminal(f"Error refreshing results: {e}")
        else:
            self.log_to_terminal(f"Results directory does not exist: {results_dir}")

    def refresh_videos(self):
        """Refresh the video list"""
        video_dir = self.video_dir.get()
        self.video_listbox.delete(0, tk.END)
        
        if os.path.exists(video_dir):
            try:
                for file in os.listdir(video_dir):
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
                        self.video_listbox.insert(tk.END, file)
            except Exception as e:
                self.log_to_terminal(f"Error refreshing videos: {e}")

    def on_file_select(self, event):
        """Handle file selection in results listbox"""
        selection = self.results_listbox.curselection()
        if selection:
            filename = self.results_listbox.get(selection[0])
            self.show_image_preview(filename)

    def show_image_preview(self, filename):
        """Show image preview (placeholder for now)"""
        try:
            # Clear previous preview
            for widget in self.preview_frame.winfo_children():
                if isinstance(widget, ttk.Label) and widget != self.preview_label:
                    widget.destroy()
            
            filepath = os.path.join(self.results_dir.get(), filename)
            if os.path.exists(filepath):
                # For now, just show the filename and path
                info_text = f"Selected: {filename}\nPath: {filepath}\nSize: {os.path.getsize(filepath)} bytes"
                preview_info = ttk.Label(self.preview_frame, text=info_text, wraplength=300)
                preview_info.pack(pady=10)
                
                # Add button to open in default viewer
                ttk.Button(self.preview_frame, text="Open in Default Viewer", 
                          command=lambda: self.open_file_externally(filepath)).pack(pady=5)
        except Exception as e:
            self.log_to_terminal(f"Error showing preview: {e}")

    def open_file_externally(self, filepath):
        """Open file in default system viewer"""
        try:
            import subprocess
            subprocess.Popen(['start', '', filepath], shell=True)
        except Exception as e:
            self.log_to_terminal(f"Error opening file: {e}")

    def open_results_folder(self):
        """Open results folder in file explorer"""
        results_dir = self.results_dir.get()
        if os.path.exists(results_dir):
            try:
                import subprocess
                subprocess.Popen(['explorer', os.path.abspath(results_dir)])
            except Exception as e:
                self.log_to_terminal(f"Error opening folder: {e}")
        else:
            self.log_to_terminal(f"Results directory does not exist: {results_dir}")

    def open_video_folder(self):
        """Open video folder in file explorer"""
        video_dir = self.video_dir.get()
        if os.path.exists(video_dir):
            try:
                import subprocess
                subprocess.Popen(['explorer', os.path.abspath(video_dir)])
            except Exception as e:
                self.log_to_terminal(f"Error opening video folder: {e}")

    def create_video_from_current(self):
        """Create video from current image directory"""
        image_dir = self.results_dir.get()
        if not os.path.exists(image_dir):
            messagebox.showerror("Error", f"Image directory does not exist: {image_dir}")
            return
        
        # Generate timestamp-based filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"analysis_{timestamp}.mp4"
        
        cmd = [
            self.python_exe, "create_vision_video.py",
            "--input-dir", image_dir,
            "--output", video_filename,
            "--fps", "10"
        ]
        
        self.run_command(cmd, f"Creating Video from {image_dir}")

    def play_video(self, event):
        """Play selected video"""
        selection = self.video_listbox.curselection()
        if selection:
            filename = self.video_listbox.get(selection[0])
            filepath = os.path.join(self.video_dir.get(), filename)
            if os.path.exists(filepath):
                self.open_file_externally(filepath)

    def run_command_with_video(self, cmd, description, image_dir=None):
        """Run command and automatically create video afterwards"""
        def run_with_video():
            # First run the original command
            self.log_to_terminal(f"\n{'='*60}")
            self.log_to_terminal(f"{description}")
            self.log_to_terminal(f"Command: {' '.join(cmd)}")
            self.log_to_terminal(f"{'='*60}")
            
            try:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, universal_newlines=True,
                    cwd=self.cwd
                )
                
                # Stream output in real-time
                for line in process.stdout:
                    self.log_to_terminal(line.rstrip())
                
                process.wait()
                
                if process.returncode == 0:
                    self.log_to_terminal(f"\n[OK] {description} completed successfully")
                    
                    # Auto-generate video if image directory is specified and has images
                    if image_dir and os.path.exists(image_dir):
                        # Check if there are actually images to make a video from
                        image_files = []
                        for ext in ['.png', '.jpg', '.jpeg']:
                            image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
                        
                        if image_files:
                            self.log_to_terminal(f"\n[INFO] Auto-generating video from {image_dir} ({len(image_files)} images)...")
                            
                            # Generate timestamp-based filename
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            video_filename = f"{description.lower().replace(' ', '_')}_{timestamp}.mp4"
                        
                        video_cmd = [
                            self.python_exe, "create_vision_video.py",
                            "--input-dir", image_dir,
                            "--output", video_filename,
                            "--fps", "10"
                        ]
                        
                        video_process = subprocess.Popen(
                            video_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, universal_newlines=True,
                            cwd=self.cwd
                        )
                        
                        for line in video_process.stdout:
                            self.log_to_terminal(line.rstrip())
                        
                        video_process.wait()
                        
                        if video_process.returncode == 0:
                            self.log_to_terminal(f"[OK] Video created successfully: {video_filename}")
                            # Refresh results and videos
                            self.root.after(100, self.refresh_results)
                        else:
                            self.log_to_terminal(f"[WARNING] Video creation failed")
                    else:
                        self.log_to_terminal(f"\n[INFO] No images found in {image_dir} - skipping video generation")
                else:
                    self.log_to_terminal(f"\n[ERROR] {description} failed with exit code {process.returncode}")
                    
            except Exception as e:
                self.log_to_terminal(f"\n[ERROR] Error running {description}: {e}")
        
        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=run_with_video)
        thread.daemon = True
        thread.start()

    def on_tab_changed(self, event):
        """Handle tab change events"""
        selected_tab = event.widget.tab('current')['text']
        if selected_tab == "Results":
            # Auto-refresh results when switching to Results tab
            self.root.after(100, self.refresh_results)

    def initialize_results(self):
        """Initialize results display on startup"""
        try:
            # Create directories if they don't exist
            os.makedirs("images/vision", exist_ok=True)
            os.makedirs("images/vision/stages", exist_ok=True)
            os.makedirs("images/vision/aeb", exist_ok=True) 
            os.makedirs("images/lidar", exist_ok=True)
            os.makedirs("videos/vision", exist_ok=True)
            
            # Initial refresh
            self.refresh_results()
        except Exception as e:
            self.log_to_terminal(f"Error initializing results: {e}")

    # Terminal control
    def clear_terminal(self):
        self.terminal_text.delete('1.0', 'end')

    def save_terminal_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Terminal Output",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialname=f"f1tenth_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.terminal_text.get('1.0', 'end'))
                messagebox.showinfo("Success", f"Output saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save output: {e}")


def main():
    root = tk.Tk()
    app = F1TenthSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()