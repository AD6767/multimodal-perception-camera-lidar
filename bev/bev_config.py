class BEVConfig:
    # Spatial range(meters)
    X_MIN = 0.0
    X_MAX = 70.4
    Y_MIN = -40.0
    Y_MAX = 40.0

    RESOLUTION = 0.1 # meters per pixel
    STRIDE: int = 4 # downsample stride in the CNN

    @property
    def HEIGHT(self):
        # forward direction
        return int((self.X_MAX - self.X_MIN) / self.RESOLUTION)
    
    @property
    def WIDTH(self):
        # left-right direction
        return int((self.Y_MAX - self.Y_MIN) / self.RESOLUTION)
    
    @property
    def OUT_HEIGHT(self):
        return self.HEIGHT // self.STRIDE

    @property
    def OUT_WIDTH(self):
        return self.WIDTH // self.STRIDE
