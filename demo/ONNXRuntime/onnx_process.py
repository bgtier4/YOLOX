args.model = <ONNX_MODEL_PATH>

session = onnxruntime.InferenceSession(args.model)

ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
output = session.run(None, ort_inputs)