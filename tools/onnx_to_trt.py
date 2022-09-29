# import pycuda.driver as cuda
# import pycuda.autoinit

# import numpy as np
# import tensorrt as trt
# # logger to capture errors, warnings, and other information during the build and inference phases
# TRT_LOGGER = trt.Logger()
# def build_engine(onnx_file_path):

#     # initialize TensorRT engine and parse ONNX model

#     builder = trt.Builder(TRT_LOGGER)
#     network = builder.create_network()

#     parser = trt.OnnxParser(network, TRT_LOGGER)
#     # parse ONNX
#     with open(onnx_file_path, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         parser.parse(model.read())
#     print('Completed parsing of ONNX file')

#     builder.max_batch_size = 1
#     if builder.platform_has_fast_fp16:
#         builder.fp16_mode = True

#     engine = builder.build_cuda_engine(network)
#     context = engine.create_execution_context()
#     return engine, context


# def main():
#     engine, context = build_engine('../models/yolox_x.onnx')

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from PIL import Image

from os import listdir

CHANNEL = 3
HEIGHT = 640
WIDTH = 640

class ImageBatchStream():
  def __init__(self, batch_size, calibration_files, preprocessor):
    self.batch_size = batch_size
    self.max_batches = (len(calibration_files) // batch_size) + \
                       (1 if (len(calibration_files) % batch_size) \
                        else 0)
    self.files = calibration_files
    self.calibration_data = np.zeros((batch_size, CHANNEL, HEIGHT, WIDTH), \
                                     dtype=np.float32)
    self.batch = 0
    self.preprocessor = preprocessor

  @staticmethod
  def read_image_chw(path):
    img = Image.open(path).resize((WIDTH,HEIGHT), Image.NEAREST)
    im = np.array(img, dtype=np.float32, order='C')
    im = im[:,:,::-1]
    im = im.transpose((2,0,1))
    return im

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
        img = self.preprocessor(img)
        imgs.append(img)
      for i in range(len(imgs)):
        self.calibration_data[i] = imgs[i]
      self.batch += 1
      return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
    else:
      return np.array([])


# class EntropyCalibrator(trt.IInt8EntropyCalibrator):
#     def __init__(self):
#         trt.IInt8EntropyCalibrator.__init__(self)


def build_engine(model_file, max_ws=512*1024*1024, fp16=False, int8=False):
    print("building engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    #builder.fp16_mode = fp16
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    elif int8:
        NUM_IMAGES_PER_BATCH = 1
        # Calibration files should overlap with training/val/test sets!!!
        #calibration_files = listdir('/home/benjamin-gilby/venv_project/YOLOX/datasets/COCO/val2017')
        
        # How to make batch stream and int8 calibrator?
        batchstream = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files, preprocessor)

        Int8_calibrator = trt.IInt8EntropyCalibrator(["input_node_name"], batchstream)

        config.flags |= 1 << int(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8_calibrator
    
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.\
                                                  EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            #last_layer = network.get_layer(network.num_layers - 1)
            #network.mark_output(last_layer.get_output(0))
            
            # engine = builder.build_engine(network, config=config)
            # return engine
            serialized_network = builder.build_serialized_network(network, config=config)
            return serialized_network
            


# engine = build_engine("models/yolox_x.onnx")

# with open('engine.trt', 'wb') as f:
#     f.write(bytearray(engine.serialize()))

serialized_network = build_engine("models/yolox_x.onnx", fp16=False, int8=True)
with open('enginefp16.trt', 'wb') as f:
     f.write(bytearray(serialized_network))