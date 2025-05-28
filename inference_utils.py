import av
import os
import pims
import numpy as np
from fractions import Fraction
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image


class VideoReader(Dataset):
    def __init__(self, path, transform=None):
        self.video = pims.PyAVVideoReader(path)
        self.rate = self.video.frame_rate
        self.transform = transform
        
    @property
    def frame_rate(self):
        # Ensure frame_rate is always numeric
        print(f"DEBUG VideoReader.frame_rate: self.rate type={type(self.rate)}, value={self.rate}")
        if isinstance(self.rate, str):
            try:
                converted = float(self.rate)
                print(f"DEBUG VideoReader.frame_rate: Converted string to float: {converted}")
                return converted
            except ValueError:
                raise ValueError(f"Invalid frame_rate: '{self.rate}' cannot be converted to float")
        print(f"DEBUG VideoReader.frame_rate: Returning numeric rate: {self.rate}")
        return self.rate
        
    def __len__(self):
        return len(self.video)
        
    def __getitem__(self, idx):
        frame = self.video[idx]
        frame = Image.fromarray(np.asarray(frame))
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=1000000):
        print(f"DEBUG VideoWriter.__init__: Starting with path={path}, frame_rate={frame_rate} (type={type(frame_rate)}), bit_rate={bit_rate}")
        
        try:
            self.container = av.open(path, mode='w')
            print(f"DEBUG VideoWriter: Successfully opened container")
        except Exception as e:
            print(f"DEBUG VideoWriter: Failed to open container: {e}")
            raise
        
        # Ensure frame_rate is numeric to prevent 'str' object has no attribute 'numerator' error
        print(f"DEBUG VideoWriter: frame_rate type={type(frame_rate)}, value={frame_rate}")
        if isinstance(frame_rate, str):
            try:
                frame_rate = float(frame_rate)
                print(f"DEBUG VideoWriter: Converted string frame_rate to float: {frame_rate}")
            except ValueError:
                raise ValueError(f"Invalid frame_rate: '{frame_rate}' cannot be converted to float")
        elif not isinstance(frame_rate, (int, float)):
            raise TypeError(f"frame_rate must be numeric, got {type(frame_rate)}")
        
        # Convert frame_rate to Fraction for older av library compatibility
        print(f"DEBUG VideoWriter: Starting Fraction conversion for frame_rate={frame_rate}")
        try:
            print(f"DEBUG VideoWriter: Importing Fraction...")
            from fractions import Fraction as FractionClass
            print(f"DEBUG VideoWriter: Fraction imported successfully")
            
            if isinstance(frame_rate, float):
                # Convert float to fraction (e.g., 59.94 -> 5994/100)
                frame_rate_fraction = FractionClass(frame_rate).limit_denominator(1000)
                print(f"DEBUG VideoWriter: Converted float {frame_rate} to fraction {frame_rate_fraction}")
            else:
                frame_rate_fraction = FractionClass(int(frame_rate), 1)
                print(f"DEBUG VideoWriter: Converted int {frame_rate} to fraction {frame_rate_fraction}")
        except Exception as e:
            print(f"DEBUG VideoWriter: Failed to convert frame_rate to fraction: {e}")
            print(f"DEBUG VideoWriter: Exception details: {type(e).__name__}: {str(e)}")
            # Fallback to original value
            frame_rate_fraction = frame_rate
            print(f"DEBUG VideoWriter: Using fallback frame_rate: {frame_rate_fraction}")
        
        print(f"DEBUG VideoWriter: About to call add_stream with rate={frame_rate_fraction} (type={type(frame_rate_fraction)})")
        try:
            self.stream = self.container.add_stream('h264', rate=frame_rate_fraction)
            print(f"DEBUG VideoWriter: Successfully created stream")
        except Exception as e:
            print(f"DEBUG VideoWriter: Failed to create stream: {e}")
            print(f"DEBUG VideoWriter: Exception type: {type(e)}")
            print(f"DEBUG VideoWriter: Attempted rate value: {frame_rate_fraction}")
            print(f"DEBUG VideoWriter: Attempted rate type: {type(frame_rate_fraction)}")
            raise
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bit_rate
    
    def write(self, frames):
        # frames: [T, C, H, W]
        self.stream.width = frames.size(3)
        self.stream.height = frames.size(2)
        if frames.size(1) == 1:
            frames = frames.repeat(1, 3, 1, 1) # convert grayscale to RGB
        frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        for t in range(frames.shape[0]):
            frame = frames[t]
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            self.container.mux(self.stream.encode(frame))
                
    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()


class ImageSequenceReader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = sorted(os.listdir(path))
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with Image.open(os.path.join(self.path, self.files[idx])) as img:
            img.load()
        if self.transform is not None:
            return self.transform(img)
        return img


class ImageSequenceWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)
    
    def write(self, frames):
        # frames: [T, C, H, W]
        for t in range(frames.shape[0]):
            to_pil_image(frames[t]).save(os.path.join(
                self.path, str(self.counter).zfill(4) + '.' + self.extension))
            self.counter += 1
            
    def close(self):
        pass
        
