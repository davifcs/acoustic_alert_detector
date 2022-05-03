from os import path
import argparse

import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16
import torch.onnx
import onnxruntime as ort
import numpy as np
from models.convolutional import CNN2D, CNN1D
from models.transformer import ViT


argparser = argparse.ArgumentParser(description='Export to ONNX format')

argparser.add_argument('-m',
                       '--model_path',
                       help='Path to PyTorch Lightning model checkpoint')

argparser.add_argument('-i',
                       '--input_size',
                       nargs='+',
                       type=int,
                       help='Input size in BCHW format')


def convert_onnx(model_path, input_size):
    model = CNN1D.load_from_checkpoint(checkpoint_path=model_path, embed_dim=16, hidden_dim=16, num_heads=2, patch_size=4,
                                     num_channels=1, num_patches=64, num_classes=2, dropout=0.2, lr=1e-3)
    model.eval()
    dummy_input = torch.randn(input_size, requires_grad=True)
    onnx_model_path = str(path.splitext(model_path)[0])+".onnx"

    torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=['input'],
                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # onnx_model = onnx.load_model(onnx_model_path)
    # onnx_model = convert_float_to_float16(onnx_model)
    # onnx.save(onnx_model, onnx_model_path)

    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    pred_data = model(dummy_input)

    np.testing.assert_allclose(pred_data.detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    _args = argparser.parse_args()

    convert_onnx(model_path=_args.model_path, input_size=_args.input_size)
