class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.prev_error = error
        return output

# Example usage:
# pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
# control = pid.update(measured_value=10, dt=1)
# print(control)