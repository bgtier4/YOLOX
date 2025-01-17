#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

import onnxruntime # NEW CHANGE
import tensorrt as trt # NEW CHANGE
import pycuda.driver as cuda
import pycuda.autoinit
import gc

import json
import sys
from os import path

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCModel

from yolox.data.datasets import COCO_CLASSES #,  test_coco
from yolox.layers.fast_coco_eval_api import COCOeval_opt
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.utils.demo_utils import demo_postprocess, multiclass_nms


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False, onnx=False, onnx_path="",
        onnx2trt=False, engine_file_path="", tvmeval=False, tuning_records="", autoscheduler=False,
        int8=False, start_time=0
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        print('onnx =', onnx)
        print('onnx2trt =', onnx2trt)
        print('tvmeval =', tvmeval)
        print('start time =', start_time)

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor

        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        input_shape = test_size
        if(input_shape == (640,640)):
            output_shape = (1, 8400, 85)
        elif(input_shape == (416,416)):
            output_shape = (1, 3549, 85)

        dtype = 'float32'
        if half:
            dtype = 'float16'
        if int8:
            dtype = 'int8'

        # setup model
        if onnx:
            model = self.setup_onnx(onnx_path)
        elif onnx2trt:
            context = self.setup_trt(engine_file_path)
        elif tvmeval: 
            module = self.setup_tvm(onnx_path, test_size, tuning_records, half, int8, autoscheduler)
        else:
            model = self.setup_base(model, half, trt_file, test_size)


        # to avoid cuda out of memory error
        print('emptying cache...')
        gc.collect()
        torch.cuda.empty_cache()


        s = []
        iters_per_s = []

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                tvm_imgs = np.expand_dims(imgs[0].cpu().numpy(), axis=0)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                # run inference
                if onnx:
                    outputs = self.get_outputs_onnx(imgs, model)
                elif onnx2trt:
                    outputs = self.get_outputs_trt(imgs, context, output_shape)
                elif tvmeval: 
                    outputs = self.get_outputs_tvm(tvm_imgs, module, output_shape, dtype)
                else:
                    outputs = model(imgs)

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                # postprocessing
                if not onnx and not onnx2trt and not tvmeval:
                    outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
                else:
                    predictions = demo_postprocess(outputs, input_shape, p6=False)[0]

                    boxes = predictions[:, :4]
                    scores = predictions[:, 4:5] * predictions[:, 5:]

                    boxes_xyxy = np.ones_like(boxes)
                    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
                    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
                    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
                    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

                    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nmsthre, score_thr=self.confthre, class_agnostic=False)
                    if dets is None:
                        outputs = []
                    else:
                        outputs = [torch.from_numpy(dets), 'cuda']

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
                    s.append(time.time()-start_time)
                    iters_per_s.append(1/(nms_end - start))

            # convert to coco format
            if not onnx and not onnx2trt and not tvmeval:
                data_list_elem, image_wise_data = self.convert_to_coco_format(
                    outputs, info_imgs, ids, return_outputs=True)
            else:
                data_list_elem, image_wise_data = self.convert_to_coco_format_onnx(
                    outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics, s, iters_per_s)
        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results


    # model setup functions
    def setup_base(self, model, half, trt_file, test_size):
        print(type(model))
        print(model)
        model = model.eval()
        if half:
            model = model.half()

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        return model


    def setup_onnx(self, onnx_path):
        assert(len(onnx_path) > 0), "Onnx model path was not specified!"
        session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        return session


    def setup_trt(self, engine_file_path):
        assert(len(engine_file_path) > 0), "Engine file path was not specified!"
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_file_path, 'rb') as f:
            print('reading ', engine_file_path,'...')
            engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            print('engine deserialized')
        context = engine.create_execution_context()
        print('engine context created')
        return context


    def setup_tvm(self, onnx_path, test_size, tuning_records, half, int8, autoscheduler):
        print('setting up tvm...')

        # prep tvm
        input_name = "images"
        shape_list = {input_name : (1, 3, test_size[0], test_size[1])}
        
        import onnx as onnx4tvm
        onnx_model = onnx4tvm.load(onnx_path)
        mod, params = relay.frontend.from_onnx(onnx_model, shape_list)

        new_tuning_records = None
        if half:
            mod = tvm.relay.transform.ToMixedPrecision(mixed_precision_type='float16')(mod)
            tvmc_model = TVMCModel(mod, params)
        elif int8: # this does not work well right now
            with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
                mod = relay.quantize.quantize(mod, params)

        target = 'cuda'
        dev = tvm.cuda(0)
        if tuning_records != None:
            print('tuning records found')
            if (autoscheduler):
                with tvm.auto_scheduler.ApplyHistoryBest(tuning_records):  
                    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                        lib = relay.build(mod, target=target, params=params)
            else:
                with tvm.autotvm.apply_history_best(tuning_records):  
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build(mod, target=target, params=params)
        else:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)

        module = graph_executor.GraphModule(lib["default"](dev))

        return module

    # model inference functions
    def get_outputs_onnx(self, imgs, model):
        ort_imgs = imgs[0].cpu().numpy()
        ort_imgs = {model.get_inputs()[0].name: ort_imgs[None, :, :, :]}
        
        return model.run(None, ort_imgs)[0]

    
    def get_outputs_trt(self, imgs, context, output_shape):
        # Input and output buffer okay?
        input_buffer = np.ascontiguousarray(imgs.cpu())
        output_buffer = torch.zeros(output_shape).cpu().detach().numpy()

        imgs_memory = cuda.mem_alloc(input_buffer.nbytes)
        output_memory = cuda.mem_alloc(output_buffer.nbytes)
        bindings = [int(imgs_memory), int(output_memory)]

        stream = cuda.Stream()

        # Transfer input data from python buffers to device(GPU)
        cuda.memcpy_htod_async(imgs_memory, input_buffer, stream)

        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)

        stream.synchronize()

        return output_buffer


    def get_outputs_tvm(self, tvm_imgs, module, output_shape, dtype):
        dev = tvm.cuda(0)
        input_name = "images"

        module.set_input(input_name, tvm_imgs)
        module.run()
        

        return module.get_output(0, tvm.nd.empty(output_shape, dtype, device=dev)).numpy().astype(np.float32)


    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list


    def convert_to_coco_format_onnx(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 5]
            scores = output[:, 4]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)
            
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list


    def evaluate_prediction(self, data_dict, statistics, s=None, iters_per_s=None):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        # add graphing here:
        if s and iters_per_s:
            import matplotlib.pyplot as plt
            plt.plot(s, iters_per_s)
            plt.xlabel('t (seconds)')
            plt.ylabel('iterations')
            plt.title('Inference speed over time')
            plt.legend()
            plt.show()
            plt.savefig('graphs/infer_out.png')

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")
            print('running COCOeval()')
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info