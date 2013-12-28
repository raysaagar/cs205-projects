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

  return cv2.VideoWriter(filename=output_filename,
  fourcc=int(source.get(cv.CV_CAP_PROP_FOURCC)) or DEFAULT_CODEC,
  fps=source.get(cv.CV_CAP_PROP_FPS),frameSize=(frame_width, frame_height))

def process_frame_gpu(frame, pix_to_carve):
  return seam_carve(frame, pix_to_carve)

def process_video_parallel(src, dest, new_width):
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  meta = np.int32(np.zeros(3))

  if rank == 0:
    meta[0] = int(src.get(cv.CV_CAP_PROP_FRAME_COUNT))
    meta[1] = int(src.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    meta[2] = int(src.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

  print "broadcast: video metadata"
  comm.Bcast(meta, root=0)

  old_frames = meta[0]
  frame_w = meta[1]
  frame_h = meta[2]

  pad = 0 if old_frames % size == 0 else size - old_frames % size

  new_frames = old_frames + pad
  print "calculate: buffer padding length", old_frames, new_frames

  # Buffer for video
  old_vid, new_vid = None, None

  # only do initial processing and loading on rank 0
  if rank == 0:
    # buffer for old video
    old_vid = np.uint8(np.zeros((new_frames, frame_h, frame_w, 3)))
    # buffer for new video
    new_vid = np.uint8(np.zeros((new_frames, frame_h, new_width, 3)))

    print "reading video into buffer (rank = 0)"
    curr_f = 0
    while src.grab():
      # get the next frame
      _, frame = src.retrieve()
      # store it into the buffer
      old_vid[curr_f,:,:,:] = frame
      # move to next frame
      curr_f += 1

  print "scatter: video to rank != 0"
  # receive buffer
  recvbuf = np.uint8(np.zeros((new_frames/size,frame_h, frame_w,3)))
  # scatter!
  comm.Scatter(old_vid, recvbuf,root=0)

  sendbuf = np.uint8(np.zeros((new_frames/size,frame_h, frame_w,3)))
  # process all frames in buffers 
  for i in xrange(new_frames/size):
    print "Processing frame %i (rank %i)" % (i, rank)
    sendbuf[i,:,:,:] = process_frame_gpu(recvbuf[i], frame_w - new_width)

  print "gather: video to rank 0"
  comm.Gather(sendbuf, new_vid, root=0)

  print "writing video output"
  if rank == 0:
    for i in xrange(old_frames):
      dest.write(new_vid[i])

if __name__ == '__main__':
  # P1A is required to process video

  if len(argv) != 3:
    print "Usage: python P1B.py [input video] [output video]"
  else:
    src, dest = None, None

    # set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # set up cuda
    device_id = pycuda.autoinit.device.pci_bus_id()
    node_id = MPI.Get_processor_name()
    print "Rank %d has GPU %s on Node %s" % (rank, device_id, node_id)

    new_width = 608 # 4:3 ratio

    try:
      if rank == 0:
        src = cv2.VideoCapture(argv[1])
        dest = create_video_stream(src, argv[2], frame_width=new_width)

      start = time.time()
      process_video_parallel(src, dest, new_width)
      total_time = time.time() - start

      if rank == 0:
        print "Total Processing time: %f sec" % total_time

    finally:
      # Clean up after ourselves.
      if dest: del dest
      if src: src.release()
