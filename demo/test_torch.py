# @title Base Imports {form-width: "10%"}

import io
import cv2
import matplotlib
import mediapy as media
from lolite import FacePoint
import time
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tqdm
import torch
from lolite.parsing.human_parsing.HumanParsing import HumanParsing
import sys
from locotrack_pytorch.models.locotrack_efficientnet_transformer_384_512 import LocoTrack

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "locotrack_pytorch"))

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # "cpu"
)

# human parsing类别定义（实际索引从0开始）
TSHP_PARSING_CATEGORIES = [
    {'id': 1, 'name': 'hair', 'isthing': 1, 'color': [255, 0, 255]},
    {'id': 2, 'name': 'face_skin', 'isthing': 1, 'color': [250, 219, 20]},
    {'id': 3, 'name': 'face_no_skin', 'isthing': 1, 'color': [196, 136, 136]},
    {'id': 4, 'name': 'ear', 'isthing': 1, 'color': [186, 231, 255]},
    {'id': 5, 'name': 'beard', 'isthing': 1, 'color': [89, 56, 19]},
    {'id': 6, 'name': 'neck', 'isthing': 1, 'color': [250, 140, 22]},
    {'id': 7, 'name': 'other_skin', 'isthing': 1, 'color': [147, 235, 191]},
    {'id': 8, 'name': 'arm_left', 'isthing': 1, 'color': [250, 84, 28]},
    {'id': 9, 'name': 'arm_right', 'isthing': 1, 'color': [255, 255, 0]},
    {'id': 10, 'name': 'hand_left', 'isthing': 1, 'color': [39, 107, 9]},
    {'id': 11, 'name': 'hand_right', 'isthing': 1, 'color': [21, 53, 184]},
    {'id': 12, 'name': 'leg_left', 'isthing': 1, 'color': [114, 46, 209]},
    {'id': 13, 'name': 'leg_right', 'isthing': 1, 'color': [192, 168, 231]},
    {'id': 14, 'name': 'cloth', 'isthing': 1, 'color': [45, 231, 231]},
    {'id': 15, 'name': 'belongings', 'isthing': 1, 'color': [255, 255, 255]},
]

def get_face_masks(human_parsing_model, frame, erosion_kernel_size=5, erosion_iter=1):
    """
    获取所有人的人脸区域 mask
    输入:
        frame: [H, W, C] numpy array
    返回:
        face_masks: [n, H, W] numpy array，其中 n 为人数，每个 mask 表示一个人的脸部区域
    """
    try:
        # 输出：human_instance_seg: [n, H, W], part_sem_seg: [15, H, W]
        human_instance_seg, part_sem_seg = human_parsing_model(
            frame, instance_flag=True, human_parsing_flag=True
        )
        
        # 人数
        num_people = len(human_instance_seg)
        H, W = frame.shape[:2]
        
        # 人脸相关部位类别索引（根据你的TSHP_PARSING_CATEGORIES）
        face_parts = [0,1,2,3,4]  # 只保留face相关部分
        # 利用 part_sem_seg 获取所有 face 相关的类别的区域
        # 取 face_parts 中的部分相加
        part_sem_seg = np.array(part_sem_seg)
        face_area_mask = np.sum(part_sem_seg[face_parts], axis=0).astype(np.uint8)
        total_pixels= H*W
        # 初始化每个人的 mask 列表
        face_masks = []

        # 创建腐蚀核，椭圆形或方形都可以
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))

        for i in range(num_people):
            person_mask = human_instance_seg[i]  # [H, W]，当前人的 mask

            # 当前人的脸 mask = 当前人区域 AND 脸部区域
            individual_face_mask = (person_mask*np.float32(1/255) * face_area_mask).astype(np.uint8)
            # 对 mask 做腐蚀，缩小边缘
            eroded_mask = cv2.erode(individual_face_mask, kernel, iterations=erosion_iter)
            # 二值化（阈值可以调节）
            _, binary_face_mask = cv2.threshold(eroded_mask, 127, 1, cv2.THRESH_BINARY)
            mask_ratio=np.sum(binary_face_mask)/total_pixels
            if mask_ratio>0.005:
                face_masks.append(binary_face_mask)

        if face_masks:
            face_masks = np.stack(face_masks, axis=0)  
        else:
            face_masks = None
        
        # [n, H, W]
        return face_masks

    except Exception as e:
        print(f"[get_face_masks] Error: {e}")
        return None


def plot_2d_tracks(
    video,
    points,
    visibles,
    infront_cameras=None,
    tracks_leave_trace=0,
    show_occ=False,
):
  """Visualize 2D point trajectories."""
  num_frames, num_points = points.shape[:2]

  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
  point_colors = np.zeros((num_points, 3))
  for i in range(num_points):
    point_colors[i] = (np.array(color_map(cmap_norm(i)))[:3] * 255).astype(
        np.uint8
    )

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  frames = []
  for t in range(num_frames):
    frame = video[t].copy()

    # Draw tracks on the frame
    line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
    line_infront_cameras = infront_cameras[
        max(0, t - tracks_leave_trace) : t + 1
    ]
    for s in range(line_tracks.shape[0] - 1):
      img = frame.copy()

      for i in range(num_points):
        if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
          x1, y1 = int(round(line_tracks[s, i, 0])), int(
              round(line_tracks[s, i, 1])
          )
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(
              round(line_tracks[s + 1, i, 1])
          )
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
        elif (
            show_occ
            and line_infront_cameras[s, i]
            and line_infront_cameras[s + 1, i]
        ):  # occluded
          x1, y1 = int(round(line_tracks[s, i, 0])), int(
              round(line_tracks[s, i, 1])
          )
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(
              round(line_tracks[s + 1, i, 1])
          )
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)

      alpha = (s + 1) / (line_tracks.shape[0] - 1)
      frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

    # Draw end points on the frame
    for i in range(num_points):
      if visibles[t, i]:  # visible
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 3, point_colors[i], -1, cv2.LINE_AA)
      elif show_occ and infront_cameras[t, i]:  # occluded
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 3, point_colors[i], 1, cv2.LINE_AA)

    frames.append(frame)
  frames = np.stack(frames)
  return frames

def load_video_frames(video_path):
    """
    读取视频文件并转换为模型所需的格式
    
    Args:
        video_path: 视频文件路径
    Returns:
        frames: shape为 [1, T, H, W, 3] 的数组
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(frame)
    
    cap.release()
    
    if len(frames_list) == 0:
        raise ValueError("视频中没有找到任何帧")
    
    # 转换为numpy数组: [T, H, W, 3]
    frames = np.stack(frames_list, axis=0)
    
    
    print(f"成功读取并resize视频: {len(frames_list)}帧, 分辨率: {frames.shape[1]}x{frames.shape[2]}")
    
    return frames,fps

def resize_video_normalization_and_mask(video_chunk, mask, target_size_x=512,target_size_y=384):
    """
    将视频块和关键点按长边压缩到目标尺寸
    video_chunk: [T, H, W, C]
    mask: [H, W]
    返回: resized_video, resized_mask, scale_factor, (original_h, original_w)
    """
    T, H, W, C = video_chunk.shape
    scale_factor_x = target_size_x / W
    scale_factor_y = target_size_y / H
    resized_video=np.zeros((T, target_size_y, target_size_x, C))
    for t in range(T):
        resized_video[t] = cv2.resize(video_chunk[t], (target_size_x, target_size_y))
        
    resized_video_normalized = (resized_video.astype(np.float32) / 255.0) * 2 - 1

    resized_mask = cv2.resize(mask, (target_size_x, target_size_y), interpolation=cv2.INTER_NEAREST)
    
    return resized_video_normalized, resized_mask, scale_factor_x, scale_factor_y,(H,W)



def upscale_tracks(tracks, scale_factor_x, scale_factor_y):
    """
    将跟踪结果按比例放大回原始尺寸
    tracks: [B, N,T, 2]
    scale_factor: 缩放比例
    """
    tracks[...,0]=tracks[...,0] / scale_factor_x
    tracks[...,1]=tracks[...,1] / scale_factor_y
    return tracks

def detect_face_in_chunk(face_detector, video_chunk, start_frame_idx=0):
    """
    在视频块中检测人脸关键点
    video_chunk: [T, H, W, C] numpy array
    start_frame_idx: 开始检测的帧索引（默认中间帧）
    返回: landmarks, detection_frame_idx
    """
    T = video_chunk.shape[0]
    
    # 从中间帧开始检测
    # for offset in range(T):
    #     frame_idx = start_frame_idx + offset
    #     if frame_idx >= T:
    #         frame_idx = start_frame_idx - offset
    #     if frame_idx < 0:
    #         continue
    for frame_idx in range(T):
        frame = video_chunk[frame_idx]  # [H, W, C]
        lmk_list = face_detector(frame)
        
        if len(lmk_list) > 0:
            face_points = np.array(lmk_list)  # [n_faces, 194, 2]
            return face_points, frame_idx
    
    return None, None


def select_face_masks(masks, num_face=None):
    """
    从多个人脸 mask 中选择前 num_face 个 mask。
    参数:
        masks: [n_faces, H, W]，二值 mask 图像的数组
        num_face: 选择的人脸数目 (int)，若为 None 则选择所有
    返回:
        selected_masks: [H, W] 人脸 mask
    """
    if num_face is not None:
        selected_masks = np.sum(masks[:num_face],axis=0)
    else:
        selected_masks = np.sum(masks,axis=0)
    return selected_masks

def get_face_mask_in_chunk(segmenter, video_chunk, start_frame_idx=0, erosion_kernel_size=5, erosion_iter=1):
    """
    video_chunk: [T, H, W, C] numpy array
    start_frame_idx: 开始检测的帧索引（默认中间帧）
    返回: face_mask, detection_frame_idx
    """
    T = video_chunk.shape[0]
    
    # 从中间帧开始检测
    # === 1. 向后检测 ===
    for frame_idx in range(start_frame_idx, T):
        frame = video_chunk[frame_idx]  # [H, W, C]
        face_mask = get_face_masks(segmenter, frame, erosion_kernel_size, erosion_iter)  # [N, H, W]
        if face_mask is not None:
            return face_mask, frame_idx

    # === 2. 向前检测 ===
    for frame_idx in range(start_frame_idx - 1, -1, -1):
        frame = video_chunk[frame_idx]
        face_mask = get_face_masks(segmenter, frame,erosion_kernel_size,erosion_iter)
        if face_mask is not None:
            return face_mask, frame_idx

    
    return None, None

def mask_to_query(mask, frame_idx, grid_size=32):
    """
    使用网格点与mask相乘生成query点
    mask: [H, W] - 二值mask
    frame_idx: 检测帧的索引
    grid_size: 网格密度 (默认32x32)
    返回: [N, 3] where N = 有效的网格点数量, 格式为 [frame_idx, y, x]
    """
    H, W = mask.shape
    
    # 生成均匀分布的网格点
    y_coords = np.linspace(8, H-8, grid_size).astype(np.int32)  # 避免边界
    x_coords = np.linspace(8, W-8, grid_size).astype(np.int32)
    
    # 创建网格
    ys, xs = np.meshgrid(y_coords, x_coords, indexing='ij')
    grid_ys = ys.flatten()
    grid_xs = xs.flatten()
    
    # 检查哪些网格点在mask内
    valid_mask = mask[grid_ys, grid_xs] > 0
    
    # 只保留在mask内的点
    valid_ys = grid_ys[valid_mask]
    valid_xs = grid_xs[valid_mask]
    
    if len(valid_ys) == 0:
        # 如果没有有效点，回退到使用mask中的所有点
        print("Warning: No grid points found in mask, falling back to all mask points")
        ys, xs = np.where(mask > 0)
        valid_ys, valid_xs = ys, xs
    
    # 拼成 [N, 3]: [frame_idx, y, x]
    query = np.zeros((len(valid_ys), 3), dtype=np.int32)
    query[:, 0] = frame_idx
    query[:, 1] = valid_ys
    query[:, 2] = valid_xs
    
    print(f"Generated {len(query)} query points from {grid_size}x{grid_size} grid")
    
    return query


def track_forward(model, video, query, device):
    """
    执行前向跟踪
    Args:
        model: LocoTrack模型
        video: [T, H, W, C] numpy array，已归一化的视频
        query: [N, 3] numpy array，查询点 [frame_idx, y, x]
        device: 设备
        dtype: 数据类型
    Returns:
        tracks: [N, T, 2] numpy array
        visible: [N, T] numpy array  
    """
    # 准备视频张量 [B, T, H, W, C]
    video_input = torch.from_numpy(video[None]).to(device).float()
    query_input = torch.from_numpy(query[None]).to(device).float()

    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    with torch.autocast(device_type=device, dtype=dtype):
        with torch.no_grad():
            output = model(video_input, query_input)
    # 提取结果
    tracks = output['tracks'][0].cpu().numpy()  # [N, T, 2] 
    
    # 计算可见性
    occlusion_logits = output['occlusion']
    pred_occ = torch.sigmoid(occlusion_logits)
    if 'expected_dist' in output:
        expected_dist = output['expected_dist']
        pred_occ = 1 - (1 - pred_occ) * (1 - torch.sigmoid(expected_dist))
    
    visible = (pred_occ <= 0.5)[0].cpu().numpy()  # [N, T]
    
    return tracks, visible

def process_video_chunks(video_path, 
                         segmenter, 
                         model, 
                         chunk_size=100, 
                         num_key_points=194, 
                         target_size_x=512,
                         target_size_y=384,
                         base_output_dir="output_frames_torch",
                         erosion_kernel_size=5,erosion_iter=1,
                         grid_size=32):
    """
    分块处理视频进行人脸跟踪
    """
    # 加载完整视频
    print("Loading video...")
    video_full,fps_video = load_video_frames(video_path)  # [T, H, W, C]
    T, H, W, C = video_full.shape
    print(f"Original video shape: {video_full.shape}")
    

    processed_video = []
    all_scale_factors = []  # 保存每个块的缩放比例
    chunk_timing_log=[]
    
    # 计算块的数量
    num_chunks = (T + chunk_size - 1) // chunk_size
    print(f"Processing {num_chunks} chunks of {chunk_size} frames each...")
    
    for chunk_idx in range(num_chunks):

        
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, T)
        
        print(f"Processing chunk {chunk_idx}/{num_chunks} (frames {start_idx}-{end_idx})...")
        
        # 提取当前视频块（原始分辨率）
        video_chunk_original = video_full[start_idx:end_idx]  # [chunk_T, H, W, C]
        chunk_T = video_chunk_original.shape[0]
        
        # 在当前块的中间帧检测人脸
        middle_frame_idx = chunk_T // 2
  
        detect_start_time = time.time()
        #face_mask: [N, H, W]
        face_mask, detection_frame_idx = get_face_mask_in_chunk(segmenter, video_chunk_original, middle_frame_idx,erosion_kernel_size,erosion_iter)#middle_frame_idx)
        detect_time = time.time() - detect_start_time
        
        if face_mask is None:

            print(f"No faces detected in chunk {chunk_idx+1}, saving original frames...")
            
            # 创建输出文件夹并保存原始帧
            chunk_output_dir = os.path.join(base_output_dir, f"chunk_{chunk_idx:03d}")
            os.makedirs(chunk_output_dir, exist_ok=True)
            
            for frame_idx, frame in enumerate(video_chunk_original):
                global_frame_idx = start_idx + frame_idx
                frame_filename = os.path.join(chunk_output_dir, f"frame_{global_frame_idx:04d}.png")
                # # 转换为uint8格式保存
                # frame_uint8 = ((frame + 1) / 2 * 255).astype(np.uint8)
                cv2.imwrite(frame_filename, frame)
            
            print(f"已保存 {chunk_T} 帧原始图片到: {chunk_output_dir}")

            # 记录时间日志
            chunk_timing_log.append({
                'chunk': chunk_idx + 1,
                'detection_time': detect_time,
                'tracking_time': 0.0,
                'fps': 0.0,
                'skipped': True
            })
            continue
        
        print(f"Detected {face_mask.shape[0]} faces in chunk {chunk_idx},detection_frame_idx:{detection_frame_idx}")
        # 选择关键点
        #key_face_mask: [H,W]
        key_face_mask = select_face_masks(face_mask, num_face=None)

         # 缩放视频和关键点
        video_chunk_resized_normalized, key_face_mask_resized, scale_factor_x, scale_factor_y, original_size = resize_video_normalization_and_mask(
            video_chunk_original, key_face_mask, target_size_x=int(target_size_x),target_size_y=int(target_size_y))
        all_scale_factors.append((scale_factor_x, scale_factor_y))
        # 转换为query格式
        #得到query_points: [N, 3] where 3 = [frame_idx, y, x]
        #利用mask和网格点得到query点坐标
        query = mask_to_query(key_face_mask_resized, detection_frame_idx, grid_size=grid_size)
             
        # 执行LocoTrack跟踪
        print(f"LocoTrack tracking in chunk {chunk_idx}...")
        
        track_start_time = time.time()
        
        # 使用LocoTrack进行一次性跟踪
        #tracks_numpy: [N, T, 2]
        #pred_visible_numpy: [N, T]
        tracks_numpy, pred_visible_numpy = track_forward(
            model, 
            video_chunk_resized_normalized, query, DEFAULT_DEVICE
        )
        
        tracking_time = time.time() - track_start_time
        fps = chunk_T / tracking_time if tracking_time > 0 else 0

        # 将跟踪结果放大回原始尺寸
        tracks_numpy_upscaled = upscale_tracks(tracks_numpy, scale_factor_x, scale_factor_y)

        
        print(f"  Final tracks shape: {tracks_numpy.shape}")
        print(f"  Final occluded shape: {pred_visible_numpy.shape}")
        print(f"  Original video shape: {video_chunk_original.shape}")
        
        # 绘制轨迹
        print(f"  绘制chunk {chunk_idx + 1} 的轨迹")
        video_viz = plot_2d_tracks(
            video_chunk_original, 
            tracks_numpy_upscaled.transpose(1, 0, 2),  # [N, T, 2] -> [T, N, 2]
            pred_visible_numpy.transpose(1, 0)    # [N, T] -> [T, N]
        )

        # 创建当前chunk的输出文件夹
        chunk_output_dir = os.path.join(base_output_dir, f"chunk_{chunk_idx:03d}")
        os.makedirs(chunk_output_dir, exist_ok=True)
        
        # 保存当前chunk的所有帧
        for frame_idx, frame in enumerate(video_viz):
            global_frame_idx = start_idx + frame_idx
            frame_filename = os.path.join(
                chunk_output_dir, 
                f"frame_{global_frame_idx:04d}.png"
            )
            cv2.imwrite(frame_filename, frame)
        
        print(f"  已保存 {len(video_viz)} 帧到: {chunk_output_dir}")
        
        
        # === 保存本块计时信息 ===
        chunk_timing_log.append({
            'chunk': chunk_idx ,
            'detection_time': detect_time,
            'tracking_time': tracking_time,
            'fps': fps,
            'skipped': False
        })
            
        print(f"Chunk {chunk_idx} completed")
    # 打印总结
    print("=== Timing Summary ===")
    for log in chunk_timing_log:
        if log['skipped']:
            print(f"Chunk {log['chunk']:02d}: Skipped (no face detected)")
        else:
            print(f"Chunk {log['chunk']:02d}: Detection {log['detection_time']:.3f}s | Tracking {log['tracking_time']:.3f}s | FPS {log['fps']:.2f}")
  
def load_model(ckpt_path=None, model_size='small'):

    state_dict = torch.load(ckpt_path)['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model = LocoTrack(model_size=model_size)
    model.load_state_dict(state_dict)
    model.eval()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="blackhardtest.mp4",
        help="path to a video",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of frames per chunk",
    )
    parser.add_argument(
        "--num_key_points",
        type=int,
        default=194,
        help="Number of key landmarks to track per face",
    )
    parser.add_argument(
        "--target_size_x",
        type=int,
        default=512,
        help="Target size for video resizing",
    )
    parser.add_argument(
        "--target_size_y",
        type=int,
        default=384,
        help="Target size for video resizing",
    )
    parser.add_argument(
        "--base_output_dir",
        default="output_frames_torch",
        help="Base output directory for tracking results",
    )
    parser.add_argument(
        "--erosion_kernel_size",
        type=int,
        default=5,
        help="Erosion kernel size",
    )
    parser.add_argument(
        "--erosion_iter",
        type=int,
        default=1,
        help="Erosion iterations",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=32,
        help="Grid size for generating query points (e.g., 32 for 32x32 grid)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/root/group-trainee/zyp/project1_7_15/locotrack/locotrack_pytorch/efficinent_1_1_chekpoint/epoch=0-step=6000.ckpt",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Model size",
    )

    args = parser.parse_args(args=[
     "--video_path", "/root/zyp/output/lcootrack/test7.mp4",
     "--chunk_size", "200", 
     "--target_size_x", "512", 
     "--target_size_y", "384", 
     "--base_output_dir", "test_7_small_efficentnet_b5_40_64_176_mlp_input_features_854/",
     "--erosion_kernel_size", "5",
     "--erosion_iter", "1",
     "--grid_size", "32",
     "--ckpt_path", "locotrack_pytorch/small_efficentnet_b5_40_64_176_mlp_input_features_854_chekpoint/epoch=0-step=285000.ckpt",
     "--model_size", "small_efficentnet_b5_40_64_176_mlp_input_features_854",
     ])

    print("Initializing efficientnet models...")


    print("Initializing locotrack models...")
    model = load_model(ckpt_path=args.ckpt_path,model_size=args.model_size).to(DEFAULT_DEVICE)
    # model=LocoTrack(model_size="small_efficentnet_b5_40_64_176_mlp_input_features_854").to(DEFAULT_DEVICE)
    model.eval()
    # 初始化人脸检测器
    segmenter = HumanParsing()
    # 分块处理视频
    process_video_chunks(
        args.video_path, 
        segmenter, 
        model,
        args.chunk_size,
        args.num_key_points,
        args.target_size_x,
        args.target_size_y,
        args.base_output_dir,
        args.erosion_kernel_size,
        args.erosion_iter,
        args.grid_size
    )









