from os import path
import argparse

import torch.onnx
import onnxruntime as ort
import numpy as np
from models.convolutional import DSCNN, CNN2D, CNN1D
from models.deepwise import GhostNet


argparser = argparse.ArgumentParser(description='Export to ONNX format')

argparser.add_argument('-m',
                       '--model_path',
                       help='Path to PyTorch Lightning model checkpoint')

argparser.add_argument('-mt',
                       '--model_type',
                       help='Select among: CNN1D, CNN2D, DSCNN, GN')


argparser.add_argument('-i',
                       '--input_size',
                       nargs='+',
                       type=int,
                       help='Input size in BCHW format')


def convert_onnx(model_path, model_type, input_size):
    if model_type == "DSCNN":
        model = DSCNN().load_from_checkpoint(checkpoint_path=model_path)
    elif model_type == "CNN2D":
        model = CNN2D().load_from_checkpoint(checkpoint_path=model_path)
    elif model_type == "CNN1D":
        model = CNN1D().load_from_checkpoint(checkpoint_path=model_path)
    elif model_type == "GN":
        cfgs = [
            # k, t, c, SE, s
            # stage1
            [[3, 16, 16, 0, 1]],
            # stage2
            [[3, 48, 24, 0, 2]],
            [[3, 72, 24, 0, 1]],
            # stage3
            [[5, 72, 40, 0.25, 2]],
            [[5, 120, 40, 0.25, 1]],
            # stage4
            [[3, 240, 80, 0, 2]],
            [[3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
             ],
            # stage5
            [[5, 672, 160, 0.25, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
             ]
        ]
        model = GhostNet.load_from_checkpoint(checkpoint_path=model_path,
                                              cfgs=cfgs, width=0.2, learning_rate=0.01, log_path='./', patience=20)

    model.eval()
    dummy_input = torch.randn(input_size, requires_grad=True)
    onnx_model_path = str(path.splitext(model_path)[0])+".onnx"

    torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=['input'],
                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    pred_data = model(dummy_input)

    np.testing.assert_allclose(pred_data[0].detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(pred_data[1].detach().numpy(), ort_outs[1], rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    _args = argparser.parse_args()

    convert_onnx(model_path=_args.model_path, model_type=_args.model_type, input_size=_args.input_size)
