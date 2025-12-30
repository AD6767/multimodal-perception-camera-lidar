class BEVConfig:
    X_MIN = 0.0 # Spatial range(meters)
    X_MAX = 70.4
    Y_MIN = -40.0
    Y_MAX = 40.0

    RESOLUTION = 0.1 # meters per pixel

    @property
    def WIDTH(self):
        return int((self.X_MAX - self.X_MIN) / self.RESOLUTION)
    
    @property
    def HEIGHT(self):
        return int((self.Y_MAX - self.Y_MIN) / self.RESOLUTION)
