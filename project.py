
import plotly.graph_objects as go
# import cv2
from deepface import DeepFace
import numpy as np
import time
from threading import Thread

def draw_text_with_background(img, text, position, font_scale=1, thickness=2):
    """
    Draw text with a background box on the image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate box coordinates
    padding = 10
    box_coords = (
        position[0] - padding,
        position[1] + padding,
        position[0] + text_width + padding,
        position[1] - text_height - padding
    )
    
    # Draw the background box
    cv2.rectangle(img, 
                 (box_coords[0], box_coords[1]), 
                 (box_coords[2], box_coords[3]), 
                 (0, 0, 0), 
                 cv2.FILLED)
    
    # Draw the text
    cv2.putText(img, text, position, font, font_scale, (255, 255, 255), thickness)

def display_ac_temperature(mood: str):
    """
    Display an AC-style gauge chart for a given mood:
    - 'happy' → 25–26 °C
    - 'angry' → 20–21 °C
    """
    # Map moods to temperature ranges
    mood_map = {
        "happy": (25, 26),
        "angry": (20, 21),
    }
    mood_lower = mood.strip().lower()
    if mood_lower not in mood_map:
        raise ValueError("Unsupported mood. Use 'happy' or 'angry'.")

    t_min, t_max = mood_map[mood_lower]
    t_mid = (t_min + t_max) / 2

    # Build the gauge indicator
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=t_mid,
        title={'text': f"AC Temperature: {t_min}–{t_max} °C", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [15, 30], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [t_min, t_max], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': t_max
            }
        }
    ))
    fig.update_layout(height=400, margin={'t':50, 'b':0, 'l':0, 'r':0})
    fig.show()

def detect_emotion_from_webcam():
    """
    Detect emotion using webcam and DeepFace.
    Returns either 'happy' or 'angry' based on detected expression.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open webcam")
    
    print("Press 'q' to quit the camera feed")
    
    last_emotion = None
    last_switch_time = time.time()
    COOLDOWN_PERIOD = 10  # 10 seconds cooldown
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Analyze the frame
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Calculate time since last switch
            current_time = time.time()
            time_since_last_switch = current_time - last_switch_time
            time_remaining = max(0, COOLDOWN_PERIOD - time_since_last_switch)
            
            # Map DeepFace emotions to our binary happy/angry state
            current_mood = None
            if dominant_emotion in ['happy']:
                current_mood = 'happy'
            elif dominant_emotion in ['angry', 'disgust', 'fear', 'sad']:
                current_mood = 'angry'
            
            # Draw emotion text at the bottom of the frame
            text = f"Emotion: {dominant_emotion.upper()}"
            position = (width // 2 - 100, height - 60)  # Moved up to make room for countdown
            draw_text_with_background(frame, text, position, font_scale=1, thickness=2)
            
            # Draw countdown if in cooldown period
            if time_remaining > 0:
                countdown_text = f"Next change in: {time_remaining:.1f}s"
                countdown_position = (width // 2 - 100, height - 20)
                draw_text_with_background(frame, countdown_text, countdown_position, font_scale=0.8, thickness=2)
            else:
                ready_text = "Ready for change!"
                ready_position = (width // 2 - 80, height - 20)
                draw_text_with_background(frame, ready_text, ready_position, font_scale=0.8, thickness=2)
            
            # Only update if emotion has changed and cooldown period has passed
            if current_mood and current_mood != last_emotion and time_remaining == 0:
                print(f"Emotion changed from {last_emotion} to {current_mood}")
                Thread(target=display_ac_temperature, args=(current_mood,)).start()
                last_emotion = current_mood
                last_switch_time = current_time
                
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            continue
        
        # Display the frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting emotion detection from webcam...")
    try:
        detect_emotion_from_webcam()
    except KeyboardInterrupt:
        print("\nStopping the program...")
    except Exception as e:
        print(f"An error occurred: {e}")
