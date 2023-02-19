python tools/tvm_tune.py -t 8000 -p local_models/yolox_x.onnx -o tuning_records/yolox_x_fp32_8000_autoscheduler.json --autoscheduler
python tools/tvm_tune.py -t 8000 -p local_models/yolox_x.onnx -o tuning_records/yolox_x_fp16_8000_autoscheduler.json --autoscheduler --half 


python tools/tvm_tune.py -t 8000 -p local_models/yolox_l.onnx -o tuning_records/yolox_l_fp32_16000_noautoscheduler.json -r tuning_records/yolox_l_fp32_8000_noautoscheduler.json