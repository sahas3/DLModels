import iree.runtime as iree_rt
from iree.compiler import compile_str
from iree.tools import tflite as iree_tflite
import iree.compiler.tflite as iree_tflite_compile

import pathlib

import numpy as np
import tensorflow as tf
import time
import os


modelDirName = pathlib.Path('./')
modelName = 'conv2dTransposeSamePadding'

modelPath = str(modelDirName.joinpath(modelName + '.tflite'))
mlirPath = str(modelDirName.joinpath(modelName + '.mlir'))
mlirTruncatedPath = str(modelDirName.joinpath(modelName + '_truncated.mlir'))
bytecodeModule = str(modelDirName.joinpath(modelName + '_iree.vmfb'))
tfliteIR = str(modelDirName.joinpath(modelName + 'tflite.mlir'))
tosaIR = str(modelDirName.joinpath(modelName + 'tosa.mlir'))


backends = ["llvm-cpu"]
config = "local-task"

iree_tflite_compile.compile_file(
  modelPath,
  input_type="tosa",
  output_file=bytecodeModule,
  save_temp_tfl_input=tfliteIR,
  save_temp_iree_input=tosaIR,
  target_backends=backends,
  import_only=False)

config = iree_rt.Config("local-task")
context = iree_rt.SystemContext(config=config)
with open(bytecodeModule, 'rb') as f:
  vm_module = iree_rt.VmModule.from_flatbuffer(config.vm_instance, f.read())
  context.add_vm_module(vm_module)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(
    model_path= modelPath, 
    num_threads=1)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape)*10, dtype=np.float32)

args = [input_data]

invoke = context.modules.module["main"]
result = invoke(*args).to_host()[0]

print(result)

output_shape = output_details[0]['shape']

print(input_shape)
print(output_shape)


interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data_tflite = interpreter.get_tensor(output_details[0]['index'])

print()
print("TFlite output:")
print(output_data_tflite)

print()
print(np.linalg.norm(result - output_data_tflite))
