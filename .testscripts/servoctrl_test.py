# get modules from project file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from servosChainClass import servos
import time

PORT = "/dev/ttyUSB0"

mtrs = servos(PORT, num_motors=2)

pwmtest = 100

mtrs.set_pwm([pwmtest, pwmtest])

for i in range(5):
    print(f"{mtrs.read_position()=}")
    print(f"{mtrs.read_velocity()=}")
    time.sleep(0.03)

mtrs.set_pwm([0, 0])