"""
Kalman Filter fusion of IMU and GPS for localization
refer to https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-2.ipynb?create=1

Sensor usage
IMU 100 Hz / data: ax, ay
GPS 5 Hz / data: x, y
"""

import numpy as np

# initial state
x = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

# initial uncertainty
P = np.diag([   100,                            # variance of x location
                100,                            # variance of y location
                10,                             # variance of x velocity
                10,                             # variance of y velocity
                1,                              # variance of x acceleration
                1, ])                           # variance of y acceleration

dt = 0.01         # 100 Hz

# dynamic matrix
A = np.matrix( [[1.0, 0.0, dt, 0.0, 1/2.0*dt**2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 1/2.0*dt**2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

# measurement matrix

H_imu = np.matrix(  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
H_gps = np.matrix(  [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
# measurement noise covariance R
gps_noise = 0.5 ** 2    #?
imu_noise = 0.1 ** 2 #?

R = np.matrix(  [[gps_noise, 0.0, 0.0, 0.0],
                 [0.0, gps_noise, 0.0, 0.0],
                 [0.0, 0.0, imu_noise, 0.0],
                 [0.0, 0.0, 0.0, imu_noise]])

# process noise covariance matrix Q
sa = 0.001
G = np.matrix([[1/2.0*dt**2],
                             [1/2.0*dt**2],
                             [dt],
                             [dt],
                             [1.0],
                         [1.0]])
Q = G*G.T*sa**2

#Identity matrix I
I = np.eye(6)



while True:    # when sensor is running
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x = A*x
    
    # Project the error covariance ahead
    P = A*P*A.T + Q    

    # get sensor data
    data_gps = []
    data_imu = []
    
    data_gps = get_GPS()
    data_imu = get_imu()
    
    if data_gps:
        H = H_gps
    
        # Measurement Update (Correction)
        # ===============================
        # if there is a GPS Measurement

        # Compute the Kalman Gain
        S = H*P*H.T + R
        K = (P*H.T) * np.linalg.pinv(S)
    
        
        # Update the estimate via z
        Z = get_measurement_from_data(data_gps)
        y = Z - (H*x)                            # Innovation or Residual
        x = x + (K*y)
        
        # Update the error covariance
        P = (I - (K*H))*P
    if data_imu:
        H = H_imu
    
        # Measurement Update (Correction)
        # ===============================
        # if there is a GPS Measurement

        # Compute the Kalman Gain
        S = H*P*H.T + R
        K = (P*H.T) * np.linalg.pinv(S)
    
        
        # Update the estimate via z
        Z = get_measurement_from_data(data_imu)
        y = Z - (H*x)                            # Innovation or Residual
        x = x + (K*y)
        
        # Update the error covariance
        P = (I - (K*H))*P
        
    current_estimated_location = x[:2]