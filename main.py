import cv2
from gesture_detector import HandDetector
import filters

def main():
    # 1. Start the webcam stream (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=1)
    
    current_filter = "Normal"

    print("GestureCam Running! Show your gestures to change filters. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break
            
        # 2. Mirror the frame horizontally so movements feel natural
        frame = cv2.flip(frame, 1)
        
        # 3. Process the frame to find hands and get finger flags
        frame = detector.find_hands(frame, draw=True)
        fingers = detector.get_finger_status(frame)

        # 4. Map the number of open fingers to a specific filter mode
        if fingers:
            total_fingers = fingers.count(1)
            
            if total_fingers == 1:
                current_filter = "Sketch"
            elif total_fingers == 2:
                current_filter = "Thermal"
            elif total_fingers == 3:
                current_filter = "Sepia"
            elif total_fingers == 5:
                current_filter = "Normal"

        # 5. Apply the selected filter
        if current_filter == "Sketch":
            output_frame = filters.apply_sketch(frame)
        elif current_filter == "Thermal":
            output_frame = filters.apply_thermal(frame)
        elif current_filter == "Sepia":
            output_frame = filters.apply_sepia(frame)
        else:
            output_frame = frame

        # 6. Add UI text to the screen
        cv2.putText(output_frame, f"Filter: {current_filter}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 7. Render the final output window
        cv2.imshow("GestureCam Filter Studio", output_frame)

        # 8. Close window if user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Clean up resources when exiting
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
