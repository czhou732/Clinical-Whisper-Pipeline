#!/usr/bin/env python3
import sys
import time
import os
import shutil
# 1. 强制定位到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import whisper
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
INPUT_FOLDER = "./Input"
OUTPUT_FOLDER = "./Output" 
PROCESSED_FOLDER = "./Processed"
MODEL_TYPE = "medium.en" 

class WhisperHandler(FileSystemEventHandler):
    def __init__(self, model):
        self.model = model

    def on_created(self, event):
        self.process_file(event)

    # 监听移动事件（拖拽文件也能识别）
    def on_moved(self, event):
        # 移动事件有个 dest_path (目标路径)，我们需要处理这个
        if not event.is_directory:
             # 创建一个伪造的 event 对象传给处理函数
            class MockEvent:
                is_directory = False
                src_path = event.dest_path
            self.process_file(MockEvent())

    def process_file(self, event):
        if event.is_directory:
            return

        filename = event.src_path
        if not filename.endswith(('.m4a', '.mp3', '.wav', '.mp4')):
            return

        # 2. 等待文件传输完成
        time.sleep(2) 
        if not os.path.exists(filename):
            return

        print(f"\n🎧 New file detected: {filename}")
        print("   Transcribing... (This allows you to keep working)")

        try:
            # Transcribe
            result = self.model.transcribe(filename)
            
            # Save Transcript
            base_name = os.path.basename(filename)
            name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}.md")
            
            with open(output_path, "w") as f:
                f.write(f"# Transcript: {name_without_ext}\n")
                f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"**Tags:** #transcribed #voice-memo\n")
                f.write("---\n\n")
                f.write(result["text"])
            
            print(f"✅ Done! Saved to: {output_path}")

            # Move Audio
            if os.path.exists(filename):
                shutil.move(filename, os.path.join(PROCESSED_FOLDER, base_name))
                print(f"📦 Archived audio to: {PROCESSED_FOLDER}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print(f"🚀 Loading Whisper Model... (Using Python at: {sys.executable})")
    model = whisper.load_model(MODEL_TYPE)
    print(f"👀 Watching '{INPUT_FOLDER}' for audio files...")
    
    event_handler = WhisperHandler(model)
    observer = Observer()
    observer.schedule(event_handler, INPUT_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()