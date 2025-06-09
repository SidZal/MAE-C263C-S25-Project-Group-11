from dynamixel_sdk import *
from typing import List
import numpy as np

# Class suited for this project's needs, based on Dynamixel SDK example files
# Not foolproof - assumes correct use of Dynamixel motors
class servos():
    def __init__(self, port: str, dynamixel_ids: List[int] = None, num_motors: int = 0):
        """
            Servos control initialization

            :param port: port string ("COM#" on Windows, "/dev/ttyUSB#" on Linux...)
            :param dynamixel_ids: list of IDs
            :param num_motors: if ^ not provided, provide num motors and assume sequential IDs
        """

        # Motor Addresses and Constants, see https://emanual.robotis.com/docs/en/dxl/mx/mx-28-2/
        self.PROTOCOL_VERSION            = 2.0
        self.BAUDRATE                    = 57600

        self.ADDR_TORQUE_ENABLE          = 64
        self.ADDR_OPERATING_MODE         = 11
        self.ADDR_PRESENT_POSITION       = 132
        self.ADDR_PRESENT_VELOCITY       = 128
        self.ADDR_GOAL_PWM               = 100

        self.PWM_MODE = 16
        self.DXL_MINIMUM_POSITION_VALUE  = 0 
        self.DXL_MAXIMUM_POSITION_VALUE  = 4095
        self.DXL_PWM_LIMIT = 885
        self.DXL_VELOCITY_LIMIT = 230

        # Determine motor ids
        self.motor_ids = None
        if dynamixel_ids:
            # given ids
            self.motor_ids = dynamixel_ids 
        else:
            # sequential motor ids [1, 2, ..., n]
            self.motor_ids = list(range(1, num_motors+1))
        
        self.num_motors = len(self.motor_ids)

        # SDK Handlers
        self.portHandler = PortHandler(port)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        self.groupBulkRead = GroupBulkRead(self.portHandler, 
                                           self.packetHandler)
        self.groupBulkWrite = GroupBulkWrite(self.portHandler, 
                                             self.packetHandler)

        # Open port
        assert self.portHandler.openPort()
        
        # Set baudrate
        assert self.portHandler.setBaudRate(self.BAUDRATE)

        # Set operation mode
        self._set_operating_mode(self.PWM_MODE)

        # Enable torque
        self._torque_enable(1)

        print("Initialized Servo Controller")
            
    def __del__(self):
        # Disable torque
        self._torque_enable(0)

        # Return to position mode
        self._set_operating_mode(3)

        # Close port
        self.portHandler.closePort()

    # Helper function to validate all u2d2 comms
    def _validate(self, result, error = 0):
        if result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(result))
        elif error != 0:
            print("%s" % self.packetHandler.getRxPacketError(error))
        else:
            return True
        return False
    
    # Sets torque enable on all motors
    def _torque_enable(self, enable: int):
        assert enable in [0, 1]

        for i in self.motor_ids:
            result, error = self.packetHandler.write1ByteTxRx(self.portHandler, i, self.ADDR_TORQUE_ENABLE, enable)
            assert self._validate(result, error)
    
    # Sets operating mode on all motors
    def _set_operating_mode(self, mode: int):
        assert mode in [1, 3, 4, 16]
        for id in self.motor_ids:
            result, error = self.packetHandler.write1ByteTxRx(self.portHandler, id, self.ADDR_OPERATING_MODE, mode)
            assert self._validate(result, error)

    # Converts motor pos scale to radians
    def _to_radians(self, angle):
        return 2*np.pi*((angle - self.DXL_MINIMUM_POSITION_VALUE) / self.DXL_MAXIMUM_POSITION_VALUE)

    def _read_address(self, addr, size: int = 4):
        for id in self.motor_ids:
            self.groupBulkRead.addParam(id, addr, size)

        result = self.groupBulkRead.txRxPacket()
        self._validate(result)

        val = []
        for id in self.motor_ids:
            val.append(self.groupBulkRead.getData(id, addr, size))

        self.groupBulkRead.clearParam()
        
        return val

    def read_position(self):
        q = self._read_address(self.ADDR_PRESENT_POSITION)

        # Convert motor scale to radians
        for i in range(self.num_motors):
            q[i] = 2*np.pi*((q[i] - self.DXL_MINIMUM_POSITION_VALUE) / self.DXL_MAXIMUM_POSITION_VALUE) - np.pi

        return q
    
    def read_velocity(self):
        qd = self._read_address(self.ADDR_PRESENT_VELOCITY)

        # Convert ??? to radians/s
        for i in range(self.num_motors):
            # bits adn signs are weird
            if qd[i] > 0x7fffffff:
                qd[i] -= 4294967296

            # Convert from 0.229 rev/min to rad/s
            qd[i] = qd[i] * 0.229 * 2 * np.pi / 60

        return qd
    
    def set_pwm(self, pwm: list):
        for j in range(self.num_motors):
            id = self.motor_ids[j]
            iter_pwm = int(pwm[j])

            if abs(iter_pwm) > self.DXL_PWM_LIMIT:
                iter_pwm = int(np.sign(iter_pwm) * self.DXL_PWM_LIMIT)

            # Add param, convert to 2 bytes here
            self.groupBulkWrite.addParam(
                self.motor_ids[j],     # cycle through motor id
                self.ADDR_GOAL_PWM, 
                2, 
                [DXL_LOBYTE(iter_pwm), DXL_HIBYTE(iter_pwm)])
            
        result = self.groupBulkWrite.txPacket()
        self._validate(result)

        self.groupBulkWrite.clearParam()