import time
from IntentionalUnit import IntentionalUnit
import anki_vector
import cv2
import numpy as np

def main():
    args = anki_vector.util.parse_command_args() # Get command line arguments            
    with anki_vector.Robot(args.serial, enable_audio_feed=True) as robot:
        intentionalUnit = IntentionalUnit(robot)
        #robot.behavior.drive_off_charger()
        robot.motors.set_head_motor(0.0)
        robot.camera.init_camera_feed()
        robot.behavior.say_text('Connected')
        while True:
            touched = robot.touch.last_sensor_reading.is_being_touched
            originalImage = np.array(robot.camera.latest_image.raw_image)
            originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
            intentionalImage = intentionalUnit.category(originalImage)
            
            # Display
            cv2.imshow("Vector image", intentionalImage)
            #time.sleep(0.5)
            key = cv2.waitKey(1)
            if key == ord('q'):
                robot.disconnect()
                break

if __name__ == "__main__":
    main()