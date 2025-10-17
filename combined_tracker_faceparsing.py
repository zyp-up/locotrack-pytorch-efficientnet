import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import cv2
import numpy as np
import torch
import time
import pickle
from lolite.parsing.face_parsing.FaceParsing import FaceParsing
from locotrack_pytorch.models.locotrack_model import LocoTrack
import matplotlib
import torch.nn.functional as F
from lolite import FacePoint
from PyTSAI.face_tracking import FaceTrackingWrapper as FaceTracking
from PyTSAI.face_tracking import FaceFeatures
import glob

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # "cpu"
)



class CombinedFaceTracker:
    def __init__(self, 
    locotrack_efficientnet_checkpoint="locotrack_pytorch/efficinent_transformer_934_mlp_input_features_chekpoint/epoch=0-step=110000.ckpt",
    locotrack_efficientnet_model_size="small",
    face_models_path="face_models",
    erosion_kernel_size=5,
    erosion_iter=1):
        """
        初始化组合跟踪器
        
        Args:
            locotrack_efficientnet_checkpoint: LocoTrack模型检查点路径
            locotrack_efficientnet_model_size: LocoTrack模型大小
            face_models_path: 人脸模型路径
        """
        print("Initializing LocoTrack...")
        self.locotrack_model = self.load_model(ckpt_path=locotrack_efficientnet_checkpoint, model_size=locotrack_efficientnet_model_size)
        self.locotrack_model.to(DEFAULT_DEVICE)
        
        print("Initializing Human Parsing...")
        self.segmenter = FaceParsing()
        self.detector = FacePoint()
        self.face_tracker = FaceTracking()
        self.face_tracker.initialize(face_models_path)

        self.erosion_kernel_size = erosion_kernel_size
        self.erosion_iter = erosion_iter

    def load_model(self,ckpt_path=None, model_size='small'):
        state_dict = torch.load(ckpt_path)['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model = LocoTrack(model_size=model_size)
        model.load_state_dict(state_dict)
        model.eval()
        return model
        
    def detect_tracking_gaps(self, tmp_result, min_gap_length=1):
        """
        检测跟踪结果中的空白段（跟丢的帧段）
        
        Args:
            tmp_result: FaceTracking的跟踪结果
            min_gap_length: 最小空白段长度
            
        Returns:
            gaps: 空白段列表，每个元素为 (start_idx, end_idx)
        """
        gaps = []
        gap_start = None
        
        for frame_idx, frame_results in enumerate(tmp_result):
            if len(frame_results) == 0:  # 空结果
                if gap_start is None:
                    gap_start = frame_idx
            else:  # 有结果
                if gap_start is not None:
                    gap_length = frame_idx - gap_start
                    if gap_length >= min_gap_length:
                        gaps.append((gap_start, frame_idx - 1))
                    gap_start = None
        
        # 处理视频结尾的空白段
        if gap_start is not None:
            gap_length = len(tmp_result) - gap_start
            if gap_length >= min_gap_length:
                gaps.append((gap_start, len(tmp_result) - 1))
                
        return gaps


    def find_best_quality_frame(self, tmp_result, start_idx, end_idx):
        """
        根据lmk_visibility累加和找到质量最好的帧
        
        Args:
            tmp_result: FaceTracking的跟踪结果
            start_idx: 搜索开始索引
            end_idx: 搜索结束索引（包含）
            
        Returns:
            best_frame_idx: 质量最好的帧索引
            best_face_result: 该帧的最佳人脸结果（用于生成query点）
        """
        best_frame_idx = None
        best_face_result = None
        best_frame_quality_score = -1
        
        for frame_idx in range(start_idx, end_idx + 1):
            frame_results = tmp_result[frame_idx]
            if len(frame_results) == 0:
                continue
            
            # 计算整个帧的质量分数（所有人脸的可见性和）
            frame_quality_score = 0
            frame_best_face = None
            face_best_score = -1
            
            for face_result in frame_results:
                visibility_array = np.array(face_result.lmk_visibility)
                face_score = np.sum(visibility_array)
                frame_quality_score += face_score
                
                # # 同时记录该帧中质量最好的人脸（用于生成query点）,后续要修改，不是记录最好的人脸，是所有人脸
                # if face_score > face_best_score:
                #     face_best_score = face_score
                #     frame_best_face = face_result
            
            # 比较整个帧的质量分数
            if frame_quality_score > best_frame_quality_score:
                best_frame_quality_score = frame_quality_score
                best_frame_idx = frame_idx
                best_face_result = frame_results
        
        return best_frame_idx, best_face_result,best_frame_quality_score

    def get_face_masks(self, frame, erosion_kernel_size=5, erosion_iter=1, best_face_result=None):
        """
        获取人脸区域mask
        """

        faceparsing_results = []
        for face_result in best_face_result:
            face_point=np.array(face_result.lmk.reshape(-1, 2))
            faceparsing_result = self.segmenter(frame, face_point).astype(np.float32)
            skin_region = faceparsing_result[:, :, FaceParsing.kSkin]
            eye_brow_region = faceparsing_result[:, :, FaceParsing.kEyebrow]
            lip_region = faceparsing_result[:, :, FaceParsing.kLip]
            inner_region = faceparsing_result[:, :, FaceParsing.kInnerMouth]
            eye_region = faceparsing_result[:, :, FaceParsing.kEye]
            face_mask = np.clip(
                (skin_region + eye_brow_region + lip_region + inner_region + eye_region), 0, 255
            ).astype(np.uint8)
            faceparsing_results.append(face_mask)

        faceparsing_results = np.array(faceparsing_results)
        faceparsing_results = np.sum(faceparsing_results, axis=0)


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))

        eroded_mask = cv2.erode(faceparsing_results.astype(np.uint8), kernel, iterations=erosion_iter)
        _, binary_face_mask = cv2.threshold(eroded_mask, 127, 1, cv2.THRESH_BINARY)
        
        return binary_face_mask


    def get_face_mask_in_gap(self, video_gap, start_frame_idx=0, erosion_kernel_size=5, erosion_iter=1,best_face_result=None):
        """
        video_chunk: [T, H, W, C] numpy array
        start_frame_idx: 开始检测的帧索引（默认中间帧）
        返回: face_mask, detection_frame_idx
        """
        T = video_gap.shape[0]
        # ===  向后检测 ===
        for frame_idx in range(start_frame_idx, T):
            frame = video_gap[frame_idx]  # [H, W, C]
            face_mask = self.get_face_masks(frame, erosion_kernel_size, erosion_iter,best_face_result)  # [H, W]
            if face_mask is not None:
                return face_mask, frame_idx
        
        return None, None

    def mask_to_query(self, mask, best_face_result, frame_idx=0, grid_size=8):
        """
        为图中的每个人脸生成查询点。
        该方法在每个人脸框内生成一个网格，并使用输入mask筛选出有效点。

        参数:
            mask (np.ndarray): 输入的分割掩码，通常是 [H, W] 的二值图。
            best_face_result (list): 包含人脸检测结果对象的列表。
                                    每个对象应有名为 `face_box` 的属性，
                                    格式为 [x, y, w, h]。
            frame_idx (int): 当前帧的索引。
            grid_size (int): 在人脸框内生成的网格的边长。

        返回:
            np.ndarray: 一个形状为 (N, 3) 的查询点数组，
                        每一行格式为 [frame_idx, y, x]。
                        如果没有找到任何有效点，则返回一个空的同形状数组。
        """
        all_valid_queries = []  # 用于存储从每个人脸中提取的有效点

        # 遍历检测到的每一个人脸
        for face_result in best_face_result:
            face_box = face_result.face_box
            x, y, w, h = face_box.astype(np.int32)

            # 1. 边界截断处理，确保人脸框不会超出mask的范围
            x, y = max(0, x), max(0, y)
            # 计算右下角坐标
            x2, y2 = x + w, y + h
            # 截断右下角坐标
            x2 = min(mask.shape[1], x2)
            y2 = min(mask.shape[0], y2)
            # 根据截断后的坐标重新计算宽高
            w, h = x2 - x, y2 - y

            # 如果截断后宽高无效，则跳过此人脸
            if w <= 0 or h <= 0:
                continue
                
            # 2. 在人脸框内部区域生成网格点 (为避免边缘效应，可以稍微内缩)
            # 这里为了简化和鲁棒性，我们直接在整个有效框 [x:x+w, y:y+h] 内生成
            x_coords = np.linspace(x, x + w - 1, grid_size, dtype=np.int32)
            y_coords = np.linspace(y, y + h - 1, grid_size, dtype=np.int32)
            
            ys, xs = np.meshgrid(y_coords, x_coords, indexing='ij')
            grid_ys = ys.flatten()
            grid_xs = xs.flatten()
            
            # 3. 使用mask进行筛选
            # grid_ys, grid_xs 已经是全局坐标，可以直接在mask上索引
            valid_mask_indices = mask[grid_ys, grid_xs] > 0
            valid_ys = grid_ys[valid_mask_indices]
            valid_xs = grid_xs[valid_mask_indices]
            
            # 4. 回退机制：如果网格点全无效，则在人脸框区域内寻找所有有效点
            #    这种方式比在整个mask上使用 np.where 高效得多
            if len(valid_ys) == 0:
                print(f"Warning: 在人脸框 {face_box} 内没有找到有效的网格点，将使用该区域内所有有效掩码点。")
                # 仅在人脸框子区域内进行搜索
                sub_mask = mask[y:y2, x:x2]
                sub_ys, sub_xs = np.where(sub_mask > 0)
                # 将子区域坐标转换回全局坐标
                valid_ys = sub_ys + y
                valid_xs = sub_xs + x

            # 如果当前人脸找到了有效点，则创建查询并添加到列表中
            if len(valid_ys) > 0:
                query = np.zeros((len(valid_ys), 3), dtype=np.int32)
                query[:, 0] = frame_idx
                query[:, 1] = valid_ys
                query[:, 2] = valid_xs
                all_valid_queries.append(query)
                
        # 5. 将所有找到的查询点合并成一个最终的数组
        if not all_valid_queries:
            # 如果没有找到任何有效点，返回一个正确形状的空数组
            return np.empty((0, 3), dtype=np.int32)
        
        return np.vstack(all_valid_queries)
    
    def resize_video_and_mask(self, video_chunk, query, target_size_x=256, target_size_y=256):
        """
        缩放视频和mask到目标尺寸
        """
        T, H, W, C = video_chunk.shape
        scale_factor_x = target_size_x / W
        scale_factor_y = target_size_y / H
        
        resized_video = np.zeros((T, target_size_y, target_size_x, C))
        for t in range(T):
            resized_video[t] = cv2.resize(video_chunk[t], (target_size_x, target_size_y))
            
        resized_video_normalized = (resized_video.astype(np.float32) / 255.0) * 2 - 1
        # 缩放query点
        scaled_query = query.copy().astype(np.float32)
        scaled_query[:, 1] = query[:, 1] * scale_factor_y  # y坐标
        scaled_query[:, 2] = query[:, 2] * scale_factor_x  # x坐标
        
        return resized_video_normalized, scaled_query, scale_factor_x, scale_factor_y, (H, W)
    
    def upscale_tracks(self, tracks, scale_factor_x, scale_factor_y):
        """
        将跟踪结果放大回原始尺寸
        """
        tracks[..., 0] = tracks[..., 0] / scale_factor_x
        tracks[..., 1] = tracks[..., 1] / scale_factor_y
        return tracks
    
    def track_forward(self, video, query):
        """
            执行前向跟踪
            Args:
                model: Locotrack模型
                video: [T, H, W, C] numpy array，已归一化的视频
                query: [N, 3] numpy array，查询点 [frame_idx, y, x]
                device: 设备
                radius: 跟踪半径
                threshold: 阈值
                use_certainty: 是否使用确定性
            Returns:
                tracks: [B, N, T, 2] torch tensor
                visible: [B, N, T] torch tensor  
                track_logits: [B, N, T, 2] torch tensor
                visible_logits: [B, N, T] torch tensor
        """
        video_input = torch.from_numpy(video[None]).to(DEFAULT_DEVICE).float()
        query_input = torch.from_numpy(query[None]).to(DEFAULT_DEVICE).float()
        
        dtype = torch.bfloat16 if DEFAULT_DEVICE == "cuda" else torch.float16
        with torch.autocast(device_type=DEFAULT_DEVICE, dtype=dtype):
            with torch.no_grad():
                output = self.locotrack_model(video_input, query_input)
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

    def get_box_from_points(self, points,visibles,padding=0.1):
        """
        为所有可见的点生成包围框
        
        参数:
        points: np.array, shape [N, 2] - 所有点的坐标 (x, y)
        visibles: np.array, shape [N,] - 可见性标记 (True/False 或 1/0)
        padding: float - 包围框的padding比例，默认0.1 (10%)
        
        返回:
        bbox: dict - 包含 'xmin', 'xmax', 'ymin', 'ymax', 'width', 'height', 'center'
        """
        # 确保visibles是布尔类型
        visibles = np.array(visibles, dtype=bool)
        
        # 筛选出可见的点
        visible_points = points[visibles]
        
        if len(visible_points) == 0:
            print("警告: 没有可见的点")
            return None
        
        # 计算边界
        xmin, ymin = np.min(visible_points, axis=0)
        xmax, ymax = np.max(visible_points, axis=0)
        
        # 计算宽高和中心
        width = xmax - xmin
        height = ymax - ymin
        center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
        
        # 添加padding
        if padding > 0:
            x_pad = width * padding / 2
            y_pad = height * padding / 2
            xmin -= x_pad
            xmax += x_pad
            ymin -= y_pad
            ymax += y_pad
            width = xmax - xmin
            height = ymax - ymin
        
        bbox = {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'width': width,
            'height': height,
            'center': center
        }
        
        return bbox
    def process_gap_with_locotack(self, frames, tmp_result, gap_start, gap_end, buffer=10, target_size_x=256, target_size_y=256, grid_size=8):
        """
        使用Locotrack处理跟丢的帧段
        
        Args:
            frames: 完整视频帧列表
            gap_start: 空白段开始帧索引
            gap_end: 空白段结束帧索引
            buffer: 前后扩展的帧数
            target_size: locotrack处理的目标尺寸
            grid_size: 网格密度
            
        Returns:
            tracks_data: 包含轨迹信息的字典
        """
        print(f"Processing gap {gap_start}-{gap_end} with Locotrack...")
        
        # 确定处理范围
        total_frames = len(frames)
        
        # 在gap前的buffer帧中找到质量最好的帧
        search_start_1 = max(0, gap_start - buffer)
        if gap_start - search_start_1 >= buffer:
            search_end_1 = min(gap_start - 1, total_frames - 1)
        else:
            search_end_1 = max(gap_start - 1, 0)
        
        search_end_2 = min(total_frames - 1, gap_end + buffer)
        if search_end_2 - gap_end >= buffer:
            search_start_2 = max(0, gap_end + 1)
        else:
            search_start_2 = min(gap_end + 1, total_frames - 1)

        best_frame_idx_1, best_face_result_1, best_frame_quality_score_1 = self.find_best_quality_frame(tmp_result, search_start_1, search_end_1)
        best_frame_idx_2, best_face_result_2, best_frame_quality_score_2 = self.find_best_quality_frame(tmp_result, search_start_2, search_end_2)

        # 判断使用前十帧还是后十帧的最佳结果
        use_forward_tracking = best_frame_quality_score_1 >= best_frame_quality_score_2
        
        if use_forward_tracking:
            # 前十帧更好，使用正向跟踪
            best_frame_idx = best_frame_idx_1
            best_face_result = best_face_result_1
            tracking_start = best_frame_idx
            tracking_end = min(total_frames - 1, gap_end + buffer)
            print(f"  Using forward tracking from best frame: {best_frame_idx}")
        else:
            # 后十帧更好，使用反向跟踪
            best_frame_idx = best_frame_idx_2
            best_face_result = best_face_result_2
            tracking_start = max(0, gap_start - buffer)
            tracking_end = best_frame_idx
            print(f"  Using backward tracking from best frame: {best_frame_idx}")
        
        print(f"  Tracking range: {tracking_start}-{tracking_end}")
        
        # 提取视频块
        video_gap = np.array(frames[tracking_start:tracking_end + 1])
        
        # 如果是反向跟踪，翻转视频序列
        if not use_forward_tracking:
            video_gap = video_gap[::-1]  # 时间维度翻转
        
        # 获取人脸mask
        face_mask, mask_frame_idx_in_video = self.get_face_mask_in_gap(video_gap,0, self.erosion_kernel_size, self.erosion_iter,best_face_result)
        if face_mask is None:
            print(f"  No face detected in gap {gap_start}-{gap_end}")
            return None

        # 生成查询点
        query = self.mask_to_query(face_mask, best_face_result, mask_frame_idx_in_video, grid_size)
        
        # 缩放视频和mask
        video_resized, query_scaled, scale_x, scale_y, original_size = self.resize_video_and_mask(
            video_gap, query, target_size_x=target_size_x, target_size_y=target_size_y
        )
        
        # 执行跟踪
        with torch.amp.autocast(DEFAULT_DEVICE, dtype=torch.float16, enabled=True):
            tracks, visible = self.track_forward(video_resized, query_scaled)
        
        # 放大回原始尺寸
        tracks_upscaled = self.upscale_tracks(tracks, scale_x, scale_y)
        # 转置维度 [T, N, 2] 和 [T, N]
        tracks_result = tracks_upscaled.transpose(1, 0, 2)  # [T, N, 2]
        visible_result = visible.transpose(1, 0)   # [T, N]
        
        # 如果是反向跟踪，需要将结果再次翻转回正常时序
        if not use_forward_tracking:
            tracks_result = tracks_result[::-1]  # 时间维度翻转回来
            visible_result = visible_result[::-1]  # 时间维度翻转回来
            print("  Results reversed back to forward time order")
        
        return {
            'tracks': tracks_result,
            'visible': visible_result,
            'gap_start': tracking_start,
            'gap_end': tracking_end,
            'detection_frame_idx': best_frame_idx,
            'face_box': best_face_result,
            'original_gap_start': gap_start,  # 保存原始gap信息
            'original_gap_end': gap_end,
            'tracking_direction': 'forward' if use_forward_tracking else 'reverse'
        }

    def plot_2d_tracks(self, video, points, visibles, tracks_leave_trace=0):
        """
        绘制2D轨迹
        """
        num_frames, num_points = points.shape[:2]
        
        # 预计算颜色映射
        color_map = matplotlib.colormaps.get_cmap('hsv')
        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
        point_colors = np.zeros((num_points, 3))
        for i in range(num_points):
            point_colors[i] = (np.array(color_map(cmap_norm(i)))[:3] * 255).astype(np.uint8)
        
        frames = []
        for t in range(num_frames):
            frame = video[t].copy()
            
            # 绘制轨迹线
            line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
            line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
            
            for s in range(line_tracks.shape[0] - 1):
                img = frame.copy()
                
                for i in range(num_points):
                    if line_visibles[s, i] and line_visibles[s + 1, i]:
                        x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                        x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                        cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
                
                alpha = (s + 1) / (line_tracks.shape[0] - 1)
                frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)
            
            # 绘制端点
            for i in range(num_points):
                if visibles[t, i]:
                    x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                    cv2.circle(frame, (x, y), 3, point_colors[i], -1, cv2.LINE_AA)

            frames.append(frame)
        
        return np.stack(frames)
    
    def draw_face_tracking_results(self, frame, frame_results):
        """
        绘制FaceTracking的结果（从sample.py移植）
        """
        annotated_frame = frame.copy()
        
        for result in frame_results:
            # 获取人脸边界框
            face_bbox = result.face_box
            x, y, w, h = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
            
            # 获取人脸信息
            face_id = result.face_id
            age = result.age
            gender = result.gender
            race = result.race
            confidence = result.confidence
            lmk = np.array(result.lmk).reshape(-1, 2)
            vis = np.array(result.lmk_visibility).reshape(-1, 1)
            
            # 绘制关键点
            for i, l in enumerate(lmk):
                if vis[i] > 0.9:
                    cv2.circle(annotated_frame, (int(l[0]), int(l[1])), 2, (0, 0, 255), -1)
                else:
                    cv2.circle(annotated_frame, (int(l[0]), int(l[1])), 2, (0, 255, 0), 1)
            
            # 绘制人脸框
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制文本信息
            text_info = f"ID:{face_id} Age:{age:.1f} {gender} {race}"
            text_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x
            text_y = max(y - 10, text_size[1] + 5)
            
            # 绘制文本背景
            cv2.rectangle(annotated_frame, 
                         (text_x, text_y - text_size[1] - 5), 
                         (text_x + text_size[0], text_y + 5), 
                         (0, 0, 0), -1)
            
            # 绘制文本
            cv2.putText(annotated_frame, text_info, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self, video_path, output_path, buffer_frames=10, target_size_x=256, target_size_y=256):
        """
        处理完整视频，结合两个跟踪器
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            tmp_result_path: 预处理的FaceTracking结果pickle文件路径
            buffer_frames: 跟丢帧段的前后扩展帧数
        """
        print(f"Processing video: {video_path}")
        
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25  # 默认帧率
            print(f"Warning: Could not get FPS from video, using default: {fps}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        print(f"Loaded {len(frames)} frames")

        # 创建输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process frames to get face tracking results
        tmp_result = self.face_tracker.process_video(np.array(frames))
        
        if not any(tmp_result):
            print("No face tracking results")
            for frame in frames:
                out.write(frame)
            out.release()
            return

        # 检测跟踪空白段
        gaps = self.detect_tracking_gaps(tmp_result)
        print(f"Detected {len(gaps)} tracking gaps: {gaps}")
        

        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 处理每个空白段
        gap_tracks_data = {}
        for gap_start, gap_end in gaps:
            tracks_data = self.process_gap_with_locotack(
                frames, tmp_result, gap_start, gap_end, buffer=buffer_frames, target_size_x=target_size_x, target_size_y=target_size_y
            )
            if tracks_data is not None:
                gap_tracks_data[(tracks_data['gap_start'], tracks_data['gap_end'])] = tracks_data
        
        # 生成最终视频
        print("Generating final video...")
        for frame_idx, (frame, frame_results) in enumerate(zip(frames, tmp_result)):
            
            # 检查当前帧是否在跟丢段内
            in_gap = False
            current_gap_data = None
            
            for (gap_start, gap_end), tracks_data in gap_tracks_data.items():
                if gap_start <= frame_idx <= gap_end:
                    in_gap = True
                    current_gap_data = tracks_data
                    break
            
            if not in_gap:
                # 有FaceTracking结果，使用原始绘制方式
                annotated_frame = self.draw_face_tracking_results(frame, frame_results)
            elif in_gap and current_gap_data is not None:
                # 在跟丢段内，使用locotrack轨迹
                gap_start = current_gap_data['gap_start']
                local_frame_idx = frame_idx - gap_start
                
                # if 0 <= local_frame_idx < current_gap_data['tracks'].shape[0]:
                # 创建单帧视频用于绘制轨迹
                bbox=self.get_box_from_points(current_gap_data['tracks'][local_frame_idx],current_gap_data['visible'][local_frame_idx])
                
                if bbox is not None:
                    # 创建单帧视频用于绘制轨迹
                    single_frame_tracks = current_gap_data['tracks'][local_frame_idx:local_frame_idx+1]
                    single_frame_visible = current_gap_data['visible'][local_frame_idx:local_frame_idx+1] 
                    track_frames = self.plot_2d_tracks(
                        np.array([frame]), 
                        single_frame_tracks, 
                        single_frame_visible
                    )
                    annotated_frame = track_frames[0]
                    
                    if len(frame_results) > 0:
                        cur_feat=frame_results[0].face_feature
                    else:
                        cur_feat=FaceFeatures()
                    # cur_feat = self.face_tracker.get_face_features(frame_idx)
                    bbox_list = [bbox['xmin'], bbox['ymin'], bbox['width'], bbox['height']]
                    cv2.rectangle(annotated_frame, (int(bbox_list[0]), int(bbox_list[1])), (int(bbox_list[0]+bbox_list[2]), int(bbox_list[1]+bbox_list[3])), (0,0, 255), 2)
                    new_feat = self.face_tracker.manual_fa_from_box(frame, bbox_list, cur_feat)
                    
                    cur_lmk = new_feat.getFaceLandmark(0)
                    # cur_lmk_xy = np.array(cur_lmk).reshape(-1, 2)
                    # cur_lmk_vis = new_feat.getFaceLandmarkVisibility(0)
                    cur_lmk_xy_box=new_feat.getFaceBox(0)
                    
                    cv2.rectangle(annotated_frame, (int(cur_lmk_xy_box[0]), int(cur_lmk_xy_box[1])), (int(cur_lmk_xy_box[0]+cur_lmk_xy_box[2]), int(cur_lmk_xy_box[1]+cur_lmk_xy_box[3])), (255,0, 0), 2)
                else:
                    annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()
            out.write(annotated_frame)
            


            if frame_idx % 100 == 0:
                print(f"Processed frame {frame_idx}/{len(frames)}")
        
        out.release()

        torch.cuda.empty_cache()

        print(f"Output video saved to: {output_path}")
        print("Video processing completed!")

    def batch_process_videos(self,input_base_path,output_base_path,buffer_frames=10, target_size_x=256, target_size_y=256):
        # 输入和输出路径
        # input_base_path = "/root/group-trainee/zyp/场景切割&人脸跟踪-标注/"
        # output_base_path = "./combined_locotrack_faceparsing_output"
        
        # 创建输出根目录
        os.makedirs(output_base_path, exist_ok=True)
        
        # 查找所有视频文件（支持常见视频格式）
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
        video_files = []
        
        for ext in video_extensions:
            pattern = os.path.join(input_base_path, "**", ext)
            video_files.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(video_files)} video files to process")
        
        begin_flag=False
        specific_videos_to_process = [
#         "/root/group-trainee/zyp/场景切割&人脸跟踪-标注/group3/广告/5快速移动 运动模糊/5快速移动 运动模糊10.mov"
# #     "/root/group-trainee/zyp/pyscenedetec_seg/group3/运动/10仰头侧脸 遮挡/10仰头侧脸 遮挡_scene_001.mp4",
# #     "/root/group-trainee/zyp/pyscenedetec_seg/group3/运动/4遮挡/4遮挡_scene_001.mp4",
# #     "/root/group-trainee/zyp/pyscenedetec_seg/group3/运动/12侧脸 遮挡/12侧脸 遮挡_scene_001.mp4",
# #     "/root/group-trainee/zyp/pyscenedetec_seg/group2/不同人种/手遮挡2/手遮挡2_scene_001.mp4",
# #     "/root/group-trainee/zyp/pyscenedetec_seg/group2/不同人种/4327193-uhd_2160_4096_25fps/4327193-uhd_2160_4096_25fps_scene_001.mp4",
# #     "/root/group-trainee/zyp/pyscenedetec_seg/group2/不同人种/模糊/模糊_scene_001.mp4"
        # "/root/group-trainee/zyp/场景切割&人脸跟踪-标注/group4/不同人种/5645653-uhd_4096_2160_25fps.mp4",
        # "/root/group-trainee/zyp/场景切割&人脸跟踪-标注/group3/婚礼/23多人 遮挡/23多人 遮挡1.mov",
        # "/root/zyp/output/lcootrack/combined_locotrack_efficientnet_faceparsing_output/test/badcase/10-5.mov"
        # "/root/zyp/output/lcootrack/combined_locotrack_efficientnet_faceparsing_output/test/badccase2/2阴阳脸 侧脸2.mov"
        # "/root/zyp/output/lcootrack/combined_locotrack_efficientnet_faceparsing_output/test/goodcase/模糊.mp4"
        "/root/zyp/output/lcootrack/combined_locotrack_efficientnet_faceparsing_output/test/badcase/7.远距离21.mov"
 ]
        for video_path in video_files:
            # if video_path not in specific_videos_to_process:
            #     print(f"Skipping {video_path}")
            #     continue
            # if not begin_flag:
            #     if video_path == specific_videos_to_process[-1]:
            #         begin_flag=True
            #     # if video_path in specific_videos_to_process:
            #         # begin_flag=True
            #     print(f"Skipping {video_path}")
            # #     continue
            if video_path not in specific_videos_to_process and not begin_flag:
                print(f"Skipping {video_path}")
                continue

            # begin_flag=True
            rel_path = os.path.relpath(video_path, input_base_path)
            
            # 创建对应的输出目录
            output_dir = os.path.join(output_base_path, os.path.dirname(rel_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成输出文件路径
            video_name = os.path.basename(video_path)
            output_path = os.path.join(output_dir, video_name)
            
            print(f"Processing: {video_path}")
            print(f"Output to: {output_path}")
            
            # 这里调用你的原始处理函数
            #video_path="test_og/10多人 侧脸6.mov"
            self.process_video(video_path, output_path,buffer_frames=buffer_frames, target_size_x=target_size_x, target_size_y=target_size_y)
            
            #清除显存
            torch.cuda.empty_cache()
        
        print("Batch processing completed!")

def main():
    # 配置参数
    input_base_path ="/root/zyp/output/lcootrack/combined_locotrack_efficientnet_faceparsing_output/test/"
    output_base_path = "/root/zyp/output/lcootrack/combined_locotrack_efficientnet_faceparsing_output/"
    locotrack_efficientnet_checkpoint = "locotrack_pytorch/path_to_save_checkpoints/epoch=0-step=241000.ckpt"
    locotrack_efficientnet_model_size="small"
    buffer_frames=10
    target_size_x=256
    target_size_y=256
    
    # 创建组合跟踪器
    tracker = CombinedFaceTracker(
    locotrack_efficientnet_checkpoint=locotrack_efficientnet_checkpoint,
    locotrack_efficientnet_model_size=locotrack_efficientnet_model_size,
    face_models_path="face_models",
    erosion_kernel_size=5,
    erosion_iter=1)
    
    # 处理视频
    tracker.batch_process_videos(input_base_path, output_base_path, buffer_frames=buffer_frames, target_size_x=target_size_x, target_size_y=target_size_y)

if __name__ == "__main__":
    main()
