import onnx_graphsurgeon as gs
import numpy as np
import onnx
from main import WIDTH, HEIGHT

model = onnx.load(r'C:\Users\sunpengyuan\Downloads\model.onnx')
graph = gs.import_onnx(model)

tensors = graph.tensors()

graph.inputs = [tensors["Input"].to_variable(dtype=np.float32, shape=("N", 3, HEIGHT, WIDTH))]
graph.outputs = [tensors["box"].to_variable(dtype=np.float32), tensors["cls"].to_variable(dtype=np.float32)]

graph.cleanup()

onnx.save(gs.export_onnx(graph), "./model_cut.onnx")
