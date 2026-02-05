# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performs continuous face detection with the camera.

Simply run the script and it will draw boxes around detected faces:

    python3 detect_faces.py

For more instructions, see g.co/aiy/maker
"""

from aiymakerkit import vision
import models
import time
import os

# Preventing QT errors when running without a display
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2

detector = vision.Detector(models.FACE_DETECTION_MODEL)
print(f"Loaded model: {models.FACE_DETECTION_MODEL}")
print("Starting detection loop...\n")

frame_count = 0
total_processing_time = 0.0

for frame in vision.get_frames(display=False):
    start_time = time.time()
    frame_count += 1

    faces = detector.get_objects(frame, threshold=0.1)
    vision.draw_objects(frame, faces)

    # Show the frame ourselves after drawing
    vision.draw_objects(frame, faces)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Track processing time
    frame_time = time.time() - start_time
    total_processing_time += frame_time

cv2.destroyAllWindows()
print(f"\nProcessed {frame_count} frames total")
if frame_count > 0:
    avg_time = total_processing_time / frame_count
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"Average processing time per frame: {avg_time*1000:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
