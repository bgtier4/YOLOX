import onnx
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
from tvm import relay

def make_parser():
    parser = argparse.ArgumentParser("YOLOX TVM Tune")
    parser.add_argument("--tsize", default=(640,640), type=int, help="input img size")
    parser.add_argument(
        "--half",
        dest="half",
        default=False,
        action="store_true",
        help="tune mixed precision model"
    )
    parser.add_argument(
        "--autoscheduler",
        dest="autoscheduler",
        default=False,
        action="store_true",
        help="TVM records were built with autoscheduler",
    )
    parser.add_argument("-r", "--tuning_records", default=None, type=str, help="prior tuning records")
    parser.add_argument("-p", "--onnx_path", default=None, type=str, help="path to onnx model")

    return parser


def tune_tvm(tvmc_model, prior_records, new_records, enable_autoscheduler=False):
        target = 'cuda'
        tuning_trials = 8000
        target_host = 'llvm'

        tvmc.tune(
            tvmc_model,
            target=target,
            trials=tuning_trials,
            tuner="xgb_knob",
            target_host=target_host,
            prior_records=prior_records,
            tuning_records=new_records,
            enable_autoscheduler=enable_autoscheduler,
        )


def main():
    args = make_parser().parse_args()

    input_name = "images"
    shape_list = {input_name : (1, 3, args.tsize[0], args.tsize[1])}

    onnx_model = onnx.load(args.onnx_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_list)

    if (args.half):
        mod = tvm.relay.transform.ToMixedPrecision(mixed_precision_type='float16')(mod)

    tvmc_model = TVMCModel(mod, params)

    new_records = "../tuning_records/new_records.json"

    tune_tvm(tvmc_model, args.tuning_records, new_records, enable_autoscheduler=args.autoscheduler)

if __name__ == "__main__":
    main()