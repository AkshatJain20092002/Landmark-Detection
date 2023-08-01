import cv2
import mediapipe as mp
from IPython.display import display
from PIL import Image as PILImage
import ipywidgets as widgets
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to perform landmark detection on the provided image path
def detect_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    pointer_locations = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                pointer_locations.append((x, y))
                cv2.circle(image, (x, y), radius=2, color=(0, 0, 0), thickness=-1)
                cv2.putText(image, f"{i+1}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)

    return image, pointer_locations

# Specify the image path here
image_path = 'C:/Users/ASUS/OneDrive/Pictures/Saved Pictures/2.jpg'

# Perform landmark detection on the specified image path
image, landmark_locations = detect_landmarks(image_path)

# Create an interactive output widget to display the image
out = widgets.Output(layout={'border': '1px solid black'})
display(out)

# Display the image
def display_image(scale_factor):
    with out:
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        pil_image = PILImage.fromarray(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
        display(pil_image)

display_image(1.0)

# Create a slider widget for zooming
zoom_slider = widgets.FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description='Zoom:')
widgets.interactive(display_image, scale_factor=zoom_slider)

# Print the pointer locations
for i, (x, y) in enumerate(landmark_locations):
    print(f"Pointer {i+1}: X={x}, Y={y}")
