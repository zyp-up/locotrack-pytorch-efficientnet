import torch
import torch.onnx 
from locotrack_pytorch.models.locotrack_model import LocoTrack

# 第1步：ONNX兼容的模型包装器
class LocoTrackONNXWrapper(torch.nn.Module):
    def __init__(self, locotrack_model):
        super().__init__()
        self.locotrack_model = locotrack_model
        
    def forward(self, video_input, query_input):
        """
        ONNX兼容的前向传播，返回元组而不是字典
        """
        # 调用原始模型，传入训练模式为False
        output_dict = self.locotrack_model(
            video=video_input, 
            query_points=query_input,
            is_training=False,  # 明确设置为推理模式
            query_chunk_size=64,
            get_query_feats=False,
            refinement_resolutions=None
        )
        
        # 提取主要输出（去掉unrefined版本，因为它们可能包含列表）
        tracks = output_dict['tracks']
        occlusion = output_dict['occlusion'] 
        expected_dist = output_dict['expected_dist']
        
        # 确保输出张量是连续的和正确的数据类型
        tracks = tracks.contiguous().float()
        occlusion = occlusion.contiguous().float()
        expected_dist = expected_dist.contiguous().float()
        
        return tracks, occlusion, expected_dist

# 第2步：更新后的 ONNX 转换函数
def Convert_ONNX(model): 
    model.eval() 
    onnx_model_wrapper = LocoTrackONNXWrapper(model)
    onnx_model_wrapper.eval()
    # 创建伪输入
    dummy_video_input = torch.randn(1,10, 256, 256, 3)
    dummy_query_input = torch.randn(1,64, 3)
    # scripted = torch.jit.trace(onnx_model_wrapper, (dummy_video_input, dummy_query_input))
    # # print(scripted.graph)
    # print("查看scripted的graph")
    # for node in scripted.graph.nodes():
    #     print(node)
    # print("=== Suspicious Nodes (Tensor -> scalar) ===")
    # for node in scripted.graph.nodes():
    #     if "prim::Int" in str(node) or "aten::item" in str(node) or "NumToTensor" in str(node):
    #         print(node)

    # 定义输入输出名称
    input_names = ['video_input', 'query_input']
    output_names = ['tracks', 'occlusion', 'expected_dist']

    # 定义动态轴
    dynamic_axes = {
        'video_input': {0: 'batch_size', 1: 'num_frames', 2: 'height', 3: 'width', 4: 'channels'},
        'query_input': {0: 'batch_size', 1: 'num_points', 2: 'tyx'},
        'tracks': {0: 'batch_size', 1: 'num_points', 2: 'num_frames',3:"points_xy"},
        'occlusion': {0: 'batch_size', 1: 'num_points', 2: 'num_frames'},
        'expected_dist': {0: 'batch_size', 1: 'num_points', 2: 'num_frames'},
    }

    # 导出模型
    torch.onnx.export(onnx_model_wrapper,
         (dummy_video_input, dummy_query_input),
         "LocoTrack_baseline.onnx",
         export_params=True,
         opset_version=20,
         do_constant_folding=True,
         input_names=input_names,
         output_names=output_names,
         dynamic_axes=dynamic_axes)
    
    print("\n模型已成功转换为 ONNX 格式，并保存为 LocoTrack_baseline.onnx")

if __name__ == "__main__": 
    # 加载您的原始 PyTorch 模型
    model_size="small"
    ckpt_path="locotrack_pytorch/path_to_save_checkpoints/epoch=0-step=241000.ckpt"
    state_dict = torch.load(ckpt_path)['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model = LocoTrack(model_size=model_size)
    model.load_state_dict(state_dict)
 
    # 转换模型为 ONNX
    Convert_ONNX(model)