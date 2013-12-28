from sys import argv
import numpy as np

import cv2
from cv2 import cv

import pycuda.autoinit
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu

DEFAULT_CODEC = cv.CV_FOURCC('P','I','M','1') # MPEG-1 codec

grayscale_kernel_source = \
"""
// Each uchar3 element is the triple (blue, green, red) with
// values \in [0..255] that represent the intensity of the
// color component in that channel.

// Use the mean of the BGR triple; this is a bit lame, but easy.
#define MEAN_BGR(triple) ((triple.x + triple.y + triple.z) / 3)

__global__ void grayscale_kernel(const uchar3* input,
                                 unsigned char* output,
                                 uint pixel_count)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < pixel_count)
       output[index] = MEAN_BGR(input[index]);
}
"""

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

def create_video_stream(source, output_filename):
  '''
  Given an input video, creates an output stream that is as similar
  to it as possible.  When no codec is detected in the input,
  default to MPG-1.
  '''
  return cv2.VideoWriter(
           filename=output_filename,
           fourcc=int(source.get(cv.CV_CAP_PROP_FOURCC)) or DEFAULT_CODEC,
           fps=source.get(cv.CV_CAP_PROP_FPS),
           frameSize=(int(source.get(cv.CV_CAP_PROP_FRAME_WIDTH)),
                      int(source.get(cv.CV_CAP_PROP_FRAME_HEIGHT))))

def process_frame_gpu(kernel, frame):
  '''
  Given a source frame, processes it on the GPU using the provided kernel
  '''
  # We take our (3-channel) source frame and create a 1-channel grayscale
  # frame of the same size.
  height, width, channels = frame.shape
  gray_channel_d = gpu.empty(shape=(height, width), dtype=np.uint8)
  frame_d = gpu.to_gpu(frame)

  kernel(frame_d, gray_channel_d, np.uint32(height * width),
         block=(768,1,1),
         grid=(int(np.ceil(gray_channel_d.size / 768.)),1))

  # Our result is a 1-channel grayscale image.  we project it over
  # each of the RGB channels.
  return np.dstack([gray_channel_d.get()] * channels)

def process_frame_cpu(frame):
  '''
  Given a source frame, processes it on the CPU
  '''
  # Here we use the mean of the BGR triple; this is a bit lame, but easy.
  frame[:] = np.dstack([frame.mean(axis=2)] * frame.shape[-1])

  # Note that we have made an (expensive) assignment here instead of
  # just stacking the result and returning.
  # This is a workaround for a bug in OpenCV, where it explodes
  # violently when presented with a frame that has a "weird" stride.
  return frame

def process_video_serial(source, destination, kernel=None, start_frame=0):
  '''
  Given a video stream, processes each frame in serial fashion.
  If a kernel is provided, each frame is processed using the GPU.
  Otherwise the CPU is used.
  '''
  # Seek our starting frame, if any specified
  if start_frame > 0:
    source.set(cv.CV_CAP_PROP_POS_FRAMES, start_frame)
    # Some video containers don't support precise frame seeking;
    # if this is the case, we bail.
    assert source.get(cv.CV_CAP_PROP_POS_FRAMES) == start_frame

  while source.grab():
    _, frame = source.retrieve()
    if kernel:
        destination.write(process_frame_gpu(kernel, frame))
    else:
        destination.write(process_frame_cpu(frame))

def process_video_parallel(source, destination, kernel):
  '''
  Process the video stream in parallel.
  '''
  raise NotImplementedError

if __name__ == '__main__':
  if len(argv) != 3:
    print 'Usage: python grayscale.py [input video] [output video]'
  else:
    destination, source = None, None

    try:
      # Open our source video and create an output stream
      source = cv2.VideoCapture(argv[1])
      destination = create_video_stream(source, argv[2])

      # Compile our kernel and execute serially
      # To use the CPU instead of the GPU, set kernel=None
      kernel = cuda_compile(grayscale_kernel_source,"grayscale_kernel")
      process_video_serial(source, destination, kernel)
    finally:
      # Clean up after ourselves.
      if destination: del destination
      if source: source.release()
