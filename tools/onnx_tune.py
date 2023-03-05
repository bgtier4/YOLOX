import argparse
import json
import logging
from distutils.util import strtobool

import onnx  # type: ignore
import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relay.frontend import from_onnx
from tvm.support import describe



def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model-name",
        type=str,
        required=True,
    )
    args.add_argument(
        "--onnx-path",
        type=str,
        required=True,
    )
    args.add_argument(
        "--input-shape",
        type=str,
        required=True,
        help='example: `[{"name": "input1", "dtype": "int64", "shape": [1, 1, 8]}]',
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--number",
        type=int,
        default=3,
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=1,
    )
    args.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
    )
    args.add_argument(
        "--adaptive-training",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        default=True,
    )
    args.add_argument(
        "--cpu-flush",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        required=True,
    )
    args.add_argument(
        "--backend",
        type=str,
        choices=["graph", "vm"],
        help="example: graph / vm",
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    parsed.input_shape = json.loads(parsed.input_shape)
    return parsed


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def main():
    describe()
    print(f"Workload: {ARGS.model_name}")

    onnx_model = onnx.load(ARGS.onnx_path)
    shape_dict = {}
    for item in ARGS.input_shape:
        print(f"  input_name : {item['name']}")
        print(f"  input_shape: {item['shape']}")
        print(f"  input_dtype: {item['dtype']}")
        shape_dict[item["name"]] = item["shape"]
    mod, params = from_onnx(onnx_model, shape_dict, freeze_params=True)


    with ms.Profiler() as profiler:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=ARGS.target,
            params=params,
            work_dir=ARGS.work_dir,
            max_trials_global=ARGS.num_trials,
            num_trials_per_iter=64,
            runner=ms.runner.LocalRunner(  # type: ignore
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=ARGS.number,
                    repeat=ARGS.repeat,
                    min_repeat_ms=ARGS.min_repeat_ms,
                    enable_cpu_cache_flush=ARGS.cpu_flush,
                ),
                alloc_repeat=1,
            ),
            cost_model=ms.cost_model.XGBModel(  # type: ignore
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=ARGS.adaptive_training,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
        )
        lib = ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=ARGS.target,
            params=params,
            backend=ARGS.backend,
        )

    print("Tuning Time:")
    print(profiler.table())


if __name__ == "__main__":
    main()
