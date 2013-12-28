from sys import argv
from mpi4py import MPI
import pycuda.autoinit
import numpy as np
import time

import cv2
from cv2 import cv

# Import seam_carve from P1A
from P1A import seam_carve

DEFAULT_CODEC = cv.CV_FOURCC('P','I','M','1') # MPEG-1 codec

def create_video_stream(source, output_filename, frame_width=None):
  '''
  Given an input video, creates an output stream that is as similar
  to it as possible.  When no codec is detected in the input,
  default to MPG-1.
  '''

  # Set frame width of output video stream
  if not frame_width:
    frame_width = int(source.get(cv.CV_CAP_PROP_FRAME_WIDTH))

  frame_height = int(source.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

  print "creating %i fps video stream of size: %i x %i" % (source.get(cv.CV_CAP_PROP_FPS), frame_width, frame_height)

  return cv2.VideoWriter(
           filename=output_filename,
           fourcc=int(source.get(cv.CV_CAP_PROP_FOURCC)) or DEFAULT_CODEC,
           fps=source.get(cv.CV_CAP_PROP_FPS),
           frameSize=(frame_width, frame_height))

def process_frame_gpu(frame, remove_pixels):
  return seam_carve(frame, remove_pixels)

def process_video_parallel(src, dest, new_width):

  # MPI setup
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()


  meta = np.int32(np.zeros(3))
  if rank == 0:
    meta[0] = int(src.get(cv.CV_CAP_PROP_FRAME_COUNT))
    meta[1] = int(src.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    meta[2] = int(src.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
  print "broadcast: metadata"
  comm.Bcast(meta, root=0)

  old_frames = meta[0]
  width = meta[1]
  height = meta[2]
  # thanks to Kevin Zhang for help with this
  pad = 0 if old_frames % size == 0 else size - old_frames % size
  new_frames = old_frames + pad
  print "calculate: buffer padding size", old_frames, new_frames

  old_video = None
  new_video = None

  # read video into buffer when rank is 0
  if rank == 0:
    # set up buffer for old video
    old_video = np.uint8(np.zeros((new_frames, height, width, 3)))
    # set up buffer for new video
    new_video = np.uint8(np.zeros((new_frames, height, new_width, 3)))

    print "read video on rank 0"
    curr_frame = 0
    while src.grab():
      # Retrieve next frame from video
      _, frame = src.retrieve()

      # Write frame to video buffer
      old_video[curr_frame,:,:,:] = frame
      curr_frame += 1

  print "scatter: waiting to send frames"
  recvbuf = np.uint8(np.zeros((new_frames / size, height, width, 3)))
  comm.Scatter(old_video, recvbuf, root=0)

  sendbuf = np.uint8(np.zeros((new_frames / size, height, new_width, 3)))
  for i in xrange(new_frames / size):
    print "Processing frame %i on rank %i" % (i, rank)
    sendbuf[i,:,:,:] = process_frame_gpu(recvbuf[i], width - new_width)

  print "gather: waiting to get frames on rank 0"
  comm.Gather(sendbuf, new_video, root=0)

  print "write video on rank 0"
  # Write video from buffer
  if rank == 0:
    for f in xrange(old_frames):
      dest.write(new_video[f])

if __name__ == '__main__':
  # Note: P1A is required to run the video processing in P1B
  if len(argv) != 3:
    print 'Usage: python P1B.py [input video] [output video]'
  else:
    dest, src = None, None

    # set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # set up cuda
    device_id = pycuda.autoinit.device.pci_bus_id()
    node_id   = MPI.Get_processor_name()
    print "Rank %d has GPU %s on Node %s" % (rank, device_id, node_id)

    new_width = 608 # 4:3 ratio

    try:
      if rank == 0:
        src = cv2.VideoCapture(argv[1])
        dest = create_video_stream(src, argv[2], frame_width=new_width)

      st = time.time()
      process_video_parallel(src, dest, new_width)
      totaltime = time.time() - st

      if rank == 0:
        print "Process Video Parallel Time: %f" % totaltime

    finally:
      # Clean up after ourselves.
      if dest: del dest
      if src: src.release()

