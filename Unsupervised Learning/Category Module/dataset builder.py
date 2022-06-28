import time
import anki_vector
import cv2
import numpy as np
from PIL import Image

def main():
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(args.serial, enable_audio_feed=True) as robot:
        #robot.behavior.drive_off_charger()
        robot.camera.init_camera_feed()
        robot.behavior.say_text('Connected')
        i = 0
        while True:
            originalImage = np.array(robot.camera.latest_image.raw_image)
            #originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(originalImage)
            im.save("C:/Users/emili/Documents/GitHub/YOLOv3 Vector implementation/VectorAC-0.1/datasets/images/image{}.jpg".format(i))

            robot.motors.set_head_motor(-1 if i % 30 <= 15 else 1)
            robot.motors.set_wheel_motors(25, -25)

            cv2.imshow("Vector image", originalImage)
            time.sleep(0.05)
            i+=1
            key = cv2.waitKey(1)
            if key == ord('q') or i >= 500:
                robot.disconnect()
                break

if __name__ == "__main__":
    main()