import time
from IntentionalUnit import IntentionalUnit
import anki_vector
import cv2
import numpy as np
from PIL import Image

def main():
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(args.serial, enable_audio_feed=True) as robot:
        intentionalUnit = IntentionalUnit(robot)
        #robot.behavior.drive_off_charger()
        robot.motors.set_head_motor(0.0)
        robot.camera.init_camera_feed()
        robot.behavior.say_text('Connected')
        i = 0
        while True:
            touched = robot.touch.last_sensor_reading.is_being_touched
            originalImage = np.array(robot.camera.latest_image.raw_image)
            #originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
            intentionalUnit.start(originalImage)
            #im = Image.fromarray(originalImage)
            #im.save("./datasets/images/image{}.jpg".format(i))

            cv2.imshow("Vector image", originalImage)
            time.sleep(0.05)
            i+=1
            key = cv2.waitKey(1)
            if key == ord('q'):
                robot.disconnect()
                break

if __name__ == "__main__":
    main()