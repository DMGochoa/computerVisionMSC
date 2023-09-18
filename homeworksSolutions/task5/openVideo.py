"""
This module implements functionality to play and control video playback.

The module allows users to play a video with options to pause, advance frame-by-frame, jump forward or backwards by 1 second, 
and exit the video playback. 

Diego A Moreno G
Msc Student in Electrical Engineering
Universidad Tecnológica de Pereira
17/09/2023
"""

import cv2
import os
import tkinter as tk

def obtener_dimensiones_pantalla():
    """Obtain screen dimensions using tkinter."""
    root = tk.Tk()
    ancho_pantalla = root.winfo_screenwidth()
    alto_pantalla = root.winfo_screenheight()
    root.destroy()
    return ancho_pantalla, alto_pantalla

# Get the current directory path
dir_actual = os.path.dirname(os.path.abspath(__file__))
# Join the current directory path with your video's relative path
ruta_video = os.path.join(dir_actual, 'examples', 'video.mp4')

# Initialize video capture with the video file
cap = cv2.VideoCapture(ruta_video)

# Retrieve video properties: frames per second and total frames
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Control variables
pausado = False
frame_actual = 0

# Get screen dimensions
ancho_pantalla, alto_pantalla = obtener_dimensiones_pantalla()

while cap.isOpened():
    # Read the current frame from video
    ret, frame = cap.read()
    
    # Check if video has ended
    if not ret:
        # Display end of video message
        cv2.putText(frame, 'Fin del Video', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(0) == ord('q'):
            break
        continue
    
    # Resize video if it exceeds screen dimensions
    h, w, _ = frame.shape
    if w > ancho_pantalla or h > alto_pantalla:
        escala_w = ancho_pantalla / w
        escala_h = alto_pantalla / h
        escala = min(escala_w, escala_h)
        dim = (int(w * escala), int(h * escala))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    
    # Display the current frame
    cv2.imshow('Video', frame)
    
    # Check for user input
    key = cv2.waitKey(1 if not pausado else -1)
    
    # Key controls
    if key == ord('q'):
        break
    elif key == ord('f'):
        frame_actual += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)
    elif key == ord('b'):
        frame_actual -= 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)
    elif key == ord('k'):
        frame_actual += fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)
    elif key == ord('m'):
        frame_actual -= fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)
    elif key != -1:  # Any other key toggles pause
        pausado = not pausado
    
    # Keep frame index within video boundaries
    frame_actual = max(0, min(frame_actual, total_frames - 1))
    if not pausado:
        frame_actual += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)

# Release video and close windows
cap.release()
cv2.destroyAllWindows()

