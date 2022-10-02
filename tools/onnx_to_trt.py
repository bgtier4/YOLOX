import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from PIL import Image
import cv2
import argparse

from os import listdir
import os

from yolox.data.data_augment import preproc as preprocess

CHANNEL = 3
HEIGHT = 640
WIDTH = 640

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Make fp16 tensorrt engine",
    )
    parser.add_argument(
        "--int8",
        dest="int8",
        default=False,
        action="store_true",
        help="Make int8 tensorrt engine",
    )
    parser.add_argument("-p", "--onnx_path", default=None, type=str, help="path to onnx model")
    return parser

# BATCH STREAM
class ImageBatchStream():
  def __init__(self, batch_size, calibration_files):
    self.batch_size = batch_size
    self.max_batches = (len(calibration_files) // batch_size) + \
                       (1 if (len(calibration_files) % batch_size) \
                        else 0)
    self.files = calibration_files
    self.calibration_data = np.zeros((batch_size, CHANNEL, HEIGHT, WIDTH), \
                                     dtype=np.float32)
    self.batch = 0
    print('running int8 calibration where total # batches will be = ', self.max_batches)
    print()

  @staticmethod
  def read_image_chw(path):
    img = cv2.imread(path)
    img, _ = preprocess(img, (WIDTH, HEIGHT))
    return img

  def reset(self):
    self.batch = 0

  def next_batch(self):
    if self.batch < self.max_batches:
      imgs = []
      files_for_batch = self.files[self.batch_size * self.batch : \
                        self.batch_size * (self.batch + 1)]
      for f in files_for_batch:
                        [self.batch_size * (self.batch + 1)]
      for f in files_for_batch:
        print("[ImageBatchStream] Processing ", f)
        img = ImageBatchStream.read_image_chw(f)
        #img = self.preprocessor(img) already preprocessed in read_image_chw
        imgs.append(img)
      for i in range(len(imgs)):
        self.calibration_data[i] = imgs[i]
      self.batch += 1
      print('batch = ', self.batch)
      return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
    else:
      return None


# ENTROPY CALIBRATOR
class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
  def __init__(self, input_layers, stream):
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.input_layers = input_layers
    self.stream = stream
    self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
    stream.reset()
    
    self.cache_file = 'calibration_cache'

  def get_batch_size(self):
    return self.stream.batch_size

  def get_batch(self, names):
    try:
        # Assume self.batches is a generator that provides batch data.
        data = self.stream.next_batch()
        cuda.memcpy_htod(self.d_input, data)

        return [int(self.d_input)]
    except:
        # When we're out of batches, we return either [] or None.
        # This signals to TensorRT that there is no calibration data remaining.
        print('out of batches')
        return []

  def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    # if os.path.exists(self.cache_file):
    #     with open(self.cache_file, "rb") as f:
    #         return f.read()
    return None

  def write_calibration_cache(self, cache):
    with open(self.cache_file, "wb") as f:
        f.write(cache)


def build_engine(model_file, max_ws=512*1024*1024, fp16=False, int8=False):
    if int8:
      print("building int8 engine")
    elif fp16:
      print("building fp16 engine")
    else:
      print("building fp32 engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    elif int8:
        NUM_IMAGES_PER_BATCH = 64

        # Should calibration files overlap with training/val/test sets?
        path2data = '/home/benjamin-gilby/venv_project/YOLOX/datasets/COCO/val2017'
        calibration_files = [os.path.join(path2data, f) for f in listdir(path2data)]

        # for speed
        #calibration_files = calibration_files[:(64*20)]
        
        batchstream = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files)
        Int8_calibrator = PythonEntropyCalibrator(["images"], batchstream)

        config.flags |= 1 << int(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8_calibrator
    
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.\
                                                  EXPLICIT_BATCH)
    print('explicit_batch = ', explicit_batch)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)

            serialized_network = builder.build_serialized_network(network, config=config)
            return serialized_network
            


args = make_parser().parse_args()
serialized_network = build_engine(args.onnx_path, fp16=args.fp16, int8=args.int8)
print('saving engine as engine.trt in working directory')
with open('engine.trt', 'wb') as f:
     f.write(bytearray(serialized_network))