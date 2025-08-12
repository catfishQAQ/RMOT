# ------------------------------------------------------------------------
# Adversarial Attack for RMOT (Referring Multi-Object Tracking)
# Modified from MOTR adversarial attack implementation
# ------------------------------------------------------------------------

from __future__ import print_function
import os
import random
import cv2
import numpy as np
import torch
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from models import build_model
from util.tool import load_model
from models.structures import Instances
from main import get_args_parser
from adv_attack import build_attacker, Attack_Scheduler

# Import RMOT specific modules
import torchvision.transforms.functional as F

# Set random seeds for reproducibility
np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Cost weights for different attack strategies
cost_weights_dict = {
    '1': {'C': 30.0, 'B': 30.0, 'G': 30.0},  # Consistent degradation → ↑ ID switches
    '2': {'C': -30.0, 'B': -30.0, 'G': -30.0},  # More ghost detections → ↑ ID switches
    '3': {'C': -30.0, 'B': 30.0, 'G': 30.0},  # Good labels + bad boxes → ↑ ID switches
    '4': {'C': 30.0, 'B': -30.0, 'G': -30.0},  # Bad labels + good boxes → ↑ ID switches
}

# Color palette for visualization
COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0),
             (210, 105, 30), (220, 20, 60), (192, 192, 192), (255, 228, 196), (50, 205, 50)]


class TransRMOT(object):
    """RMOT Tracker with adversarial attack support"""
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.active_trackers = {}
        self.inactive_trackers = {}
        self.disappeared_tracks = []

    def _remove_track(self, slot_id):
        self.inactive_trackers.pop(slot_id)
        self.disappeared_tracks.append(slot_id)

    def clear_disappeared_track(self):
        self.disappeared_tracks = []

    def update(self, dt_instances: Instances):
        self.frame_count += 1
        dt_idxes = set(dt_instances.obj_idxes.tolist())
        track_idxes = set(self.active_trackers.keys()).union(set(self.inactive_trackers.keys()))
        matched_idxes = dt_idxes.intersection(track_idxes)
        unmatched_tracker = track_idxes - matched_idxes

        for track_id in unmatched_tracker:
            if track_id in self.active_trackers:
                self.inactive_trackers[track_id] = self.active_trackers.pop(track_id)
            self.inactive_trackers[track_id].miss_one_frame()
            if self.inactive_trackers[track_id].miss > 10:
                self._remove_track(track_id)

        for i in range(len(dt_instances)):
            idx = dt_instances.obj_idxes[i]
            bbox = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i + 1]], axis=-1)
            label = dt_instances.labels[i]
            
            if label == 0:  # Visible track
                if idx in self.inactive_trackers:
                    self.active_trackers[idx] = self.inactive_trackers.pop(idx)
                if idx not in self.active_trackers:
                    from inference import Track
                    self.active_trackers[idx] = Track(idx)
                self.active_trackers[idx].update(bbox)
            elif label == 1:  # Occluded track
                if idx in self.active_trackers:
                    self.inactive_trackers[idx] = self.active_trackers.pop(idx)
                if idx not in self.inactive_trackers:
                    from inference import Track
                    self.inactive_trackers[idx] = Track(idx)
                self.inactive_trackers[idx].miss_one_frame()
                if self.inactive_trackers[idx].miss > 10:
                    self._remove_track(idx)

        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label == 0:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i + 1]], axis=-1)
                ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))


class RMOTAdversarialDetector(object):
    """Adversarial attack detector for RMOT"""
    
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(args, 'local_self_attn'):
            args.local_self_attn = False
        if not hasattr(args, 'local_self_attn_window_size'):
            args.local_self_attn_window_size = 5
        if not hasattr(args, 'gt_txt_path'):
            args.gt_txt_path = None
        # Build model and load weights
        self.model, _, _ = build_model(args)
        checkpoint_id = int(args.resume.split('/')[-1].split('.')[0].split('t')[-1]) if 'checkpoint' in args.resume else 0
        self.model = load_model(self.model, args.resume)
        self.model = self.model.cuda()
        
        # Set model mode based on adversarial flag
        if not args.adversarial:
            self.model.eval()
        else:
            self.model.adversarial = True
        
        # Load sequence information
        self.seq_num = args.seq_num.split(',') if hasattr(args, 'seq_num') else ['0005', '1.json']
        self.load_sequence_data()
        
        # Initialize tracker
        self.tr_tracker = TransRMOT()
        
        # Setup save paths
        self.setup_save_paths()
        
        # Initialize attacker if enabled
        if args.attack:
            self.setup_attacker()
        else:
            self.attacker = None
            self.attack_scheduler = None
    
    def load_sequence_data(self):
        """Load RMOT sequence data including images and language expressions"""
        # Load image list
        img_dir = os.path.join(self.args.rmot_path, 'KITTI/training/image_02', self.seq_num[0])
        img_list = os.listdir(img_dir)
        img_list = [os.path.join(img_dir, _) for _ in img_list if ('jpg' in _) or ('png' in _)]
        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)
        
        # Load language expression
        json_path = os.path.join(self.args.rmot_path, 'expression', self.seq_num[0], self.seq_num[1])
        with open(json_path, 'r') as f:
            json_info = json.load(f)
        self.json_info = json_info
        self.sentence = [json_info['sentence']]
        
        # Image preprocessing settings
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def setup_save_paths(self):
        """Setup directories for saving results"""
        exp_name = f"adv_{self.args.attack_type}_{self.args.attack_vector}" if self.args.attack else "clean"
        self.save_root = os.path.join(
            self.args.output_dir,
            f'rmot_results_{exp_name}',
            self.seq_num[0],
            self.seq_num[1].split('.')[0]
        )
        Path(self.save_root).mkdir(parents=True, exist_ok=True)
        
        self.save_img_root = os.path.join(self.save_root, 'imgs')
        Path(self.save_img_root).mkdir(parents=True, exist_ok=True)
        
        self.txt_path = os.path.join(self.save_root, 'predict.txt')
        self.vid_path = os.path.join(self.save_root, f'{self.seq_num[0]}_{self.seq_num[1].split(".")[0]}.avi')
    
    def setup_attacker(self):
        """Initialize adversarial attacker"""
        attack_func = build_attacker(self.args.attack_type)
        self.attacker = attack_func(
            self.model,
            eps=self.args.attack_eps / 255,
            alpha=self.args.attack_alpha / 255,
            steps=self.args.attack_steps,
            mean=self.mean,
            std=self.std,
            attack_type=self.args.attack_type,
            attack_vector=self.args.attack_vector if hasattr(self.args, 'attack_vector') else None,
            device=self.device,
        )
        self.attack_scheduler = Attack_Scheduler(
            self.args.attack_start,
            self.args.attack_step
        )
        
        # Setup cost weights for adversarial loss
        if hasattr(self.model, 'criterion'):
            self.model.criterion.cost_weights = cost_weights_dict[self.args.adv_weight_set]
    
    def preprocess_image(self, img_path):
        """Load and preprocess image for model input"""
        img = cv2.imread(img_path)
        assert img is not None, f"Failed to load image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_img = img.copy()
        
        # Get original dimensions
        seq_h, seq_w = img.shape[:2]
        
        # Resize and normalize
        scale = self.img_height / min(seq_h, seq_w)
        if max(seq_h, seq_w) * scale > self.img_width:
            scale = self.img_width / max(seq_h, seq_w)
        target_h = int(seq_h * scale)
        target_w = int(seq_w * scale)
        
        img = cv2.resize(img, (target_w, target_h))
        img_tensor = F.normalize(F.to_tensor(img), self.mean, self.std)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, ori_img, (seq_h, seq_w)
    
    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        return dt_instances[keep]
    
    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]
    
    @staticmethod
    def filter_dt_by_ref_scores(dt_instances: Instances, ref_threshold: float) -> Instances:
        if dt_instances.has('refers'):
            keep = dt_instances.refers > ref_threshold
            return dt_instances[keep]
        return dt_instances
    
    def visualize_and_save(self, img, dt_instances, frame_id, videowriter=None):
        """Visualize detection results and save"""
        img_show = img.copy()
        
        # Draw bounding boxes
        for i in range(len(dt_instances)):
            box = dt_instances.boxes[i].cpu().numpy()
            obj_id = dt_instances.obj_idxes[i].cpu().numpy()
            score = dt_instances.scores[i].cpu().numpy()
            
            color = COLORS_10[int(obj_id) % len(COLORS_10)]
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 2)
            
            label = f'ID:{int(obj_id)} S:{score:.2f}'
            cv2.putText(img_show, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save image
        img_path = os.path.join(self.save_img_root, f'{frame_id:06d}.jpg')
        cv2.imwrite(img_path, cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        
        # Write to video if writer provided
        if videowriter is not None:
            videowriter.write(cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        
        return img_show
    
    def write_results(self, frame_id, tracker_outputs):
        """Write tracking results to file"""
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,1,1\n'
        with open(self.txt_path, 'a') as f:
            for i in range(len(tracker_outputs)):
                x1, y1, x2, y2 = tracker_outputs[i, :4]
                w, h = x2 - x1, y2 - y1
                track_id = tracker_outputs[i, 5]
                line = save_format.format(
                    frame=int(frame_id),
                    id=int(track_id),
                    x1=x1, y1=y1, w=w, h=h
                )
                f.write(line)
    
    def run(self, prob_threshold=0.7, area_threshold=100, ref_threshold=0.5, vis=True):
        """Run adversarial attack on RMOT"""
        print(f'Processing sequence {self.seq_num}')
        print(f'Language expression: {self.sentence[0]}')
        
        # Initialize video writer
        if vis:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            videowriter = cv2.VideoWriter(self.vid_path, fourcc, 10, (self.img_width, self.img_height))
        else:
            videowriter = None
        
        # Clear previous results
        if os.path.exists(self.txt_path):
            os.remove(self.txt_path)
        
        # Initialize attack scheduler
        if self.attack_scheduler:
            self.attack_scheduler.reset()
        
        # Initialize tracking
        track_instances = None
        total_dts = 0
        total_attacked_frames = 0
        
        # Statistics collection
        stats = {
            'frame_id': [],
            'num_detections': [],
            'num_active_tracks': [],
            'attacked': []
        }
        
        # Process each frame
        for frame_idx, img_path in enumerate(tqdm(self.img_list, desc="Processing frames")):
            # Load and preprocess image
            img_tensor, ori_img, (seq_h, seq_w) = self.preprocess_image(img_path)
            
            # Apply adversarial attack if scheduled
            attacked = False
            if self.attack_scheduler and self.attack_scheduler.is_attack():
                print(f"  Attacking frame {frame_idx+1} with {self.args.attack_type} using {self.args.attack_vector}")
                self.model.adversarial = True
                img_adv = self.attacker.perturb(
                    frame_idx + 1,
                    img_tensor,
                    (seq_h, seq_w),
                    track_instances
                )
                attacked = True
                total_attacked_frames += 1
            else:
                self.model.adversarial = False
                img_adv = img_tensor
            
            # Clean up track instances for next iteration
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            
            # Model inference with language expression
            self.model.eval()
            with torch.no_grad():
                res = self.model.inference_single_image(
                    img_adv.cuda().float(),
                    self.sentence,  # Language expression for RMOT
                    (seq_h, seq_w),
                    track_instances
                )
            
            track_instances = res['track_instances']
            dt_instances = track_instances.to(torch.device('cpu'))
            
            # Detach tensors
            dt_instances.boxes = dt_instances.boxes.detach()
            dt_instances.scores = dt_instances.scores.detach()
            dt_instances.labels = dt_instances.labels.detach()
            dt_instances.obj_idxes = dt_instances.obj_idxes.detach()
            
            # Filter detections
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)
            dt_instances = self.filter_dt_by_ref_scores(dt_instances, ref_threshold)
            
            # Update tracker
            tracker_outputs = self.tr_tracker.update(dt_instances)
            
            # Collect statistics
            num_dets = len(dt_instances)
            num_active = len(self.tr_tracker.active_trackers)
            total_dts += num_dets
            
            stats['frame_id'].append(frame_idx + 1)
            stats['num_detections'].append(num_dets)
            stats['num_active_tracks'].append(num_active)
            stats['attacked'].append(attacked)
            
            # Visualization
            if vis:
                # Convert adversarial image back for visualization
                if attacked:
                    img_vis = img_adv.squeeze(0).cpu()
                    img_vis = img_vis.permute(1, 2, 0).numpy()
                    img_vis = (img_vis * np.array(self.std) + np.array(self.mean)) * 255
                    img_vis = img_vis.clip(0, 255).astype(np.uint8)
                    img_vis = cv2.resize(img_vis, (seq_w, seq_h))
                else:
                    img_vis = ori_img
                
                self.visualize_and_save(img_vis, dt_instances, frame_idx + 1, videowriter)
            
            # Write tracking results
            if len(tracker_outputs) > 0:
                self.write_results(frame_idx + 1, tracker_outputs)
            
            # Update attack scheduler
            if self.attack_scheduler:
                self.attack_scheduler.next()
        
        # Cleanup
        if videowriter:
            videowriter.release()
        
        # Save statistics
        self.save_statistics(stats, total_dts, total_attacked_frames)
        
        print(f"Processing complete. Results saved to {self.save_root}")
        print(f"Total detections: {total_dts}, Attacked frames: {total_attacked_frames}/{len(self.img_list)}")
    
    def save_statistics(self, stats, total_dts, total_attacked_frames):
        """Save tracking statistics to CSV"""
        csv_path = os.path.join(self.save_root, 'tracking_stats.csv')
        
        with open(csv_path, 'w') as f:
            f.write('frame_id,num_detections,num_active_tracks,attacked\n')
            for i in range(len(stats['frame_id'])):
                f.write(f"{stats['frame_id'][i]},{stats['num_detections'][i]},"
                       f"{stats['num_active_tracks'][i]},{int(stats['attacked'][i])}\n")
        
        # Save summary
        summary_path = os.path.join(self.save_root, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Sequence: {self.seq_num[0]} - {self.seq_num[1]}\n")
            f.write(f"Language Expression: {self.sentence[0]}\n")
            f.write(f"Total Frames: {len(self.img_list)}\n")
            f.write(f"Attacked Frames: {total_attacked_frames}\n")
            f.write(f"Total Detections: {total_dts}\n")
            f.write(f"Average Detections per Frame: {total_dts/len(self.img_list):.2f}\n")
            
            if self.args.attack:
                f.write(f"\nAttack Configuration:\n")
                f.write(f"  Type: {self.args.attack_type}\n")
                f.write(f"  Vector: {self.args.attack_vector}\n")
                f.write(f"  Epsilon: {self.args.attack_eps}\n")
                f.write(f"  Alpha: {self.args.attack_alpha}\n")
                f.write(f"  Steps: {self.args.attack_steps}\n")
                f.write(f"  Weight Set: {self.args.adv_weight_set}\n")


def main():
    """Main function to run RMOT adversarial attack"""
    parser = get_args_parser()
    
    # Add RMOT-specific arguments
    parser.add_argument('--seq_num', type=str, default='0005,1.json',
                       help='Sequence number and expression file (e.g., 0005,1.json)')
    parser.add_argument('--ref_threshold', type=float, default=0.5,
                       help='Threshold for reference expression matching score')
    
    # Adversarial attack arguments (if not already present)
    parser.add_argument('--attack', action='store_true',
                       help='Enable adversarial attack')
    parser.add_argument('--adversarial', action='store_true',
                       help='Enable adversarial mode')
    parser.add_argument('--attack_type', type=str, default='pgd',
                       choices=['pgd', 'fgsm', 'mifgsm', 'ff'],
                       help='Type of adversarial attack')
    parser.add_argument('--attack_vector', type=str, default='feat',
                       choices=['feat', 'cls', 'reg'],
                       help='Attack vector to use')
    parser.add_argument('--attack_eps', type=float, default=8,
                       help='Epsilon for adversarial attack (0-255 scale)')
    parser.add_argument('--attack_alpha', type=float, default=2,
                       help='Alpha for adversarial attack (0-255 scale)')
    parser.add_argument('--attack_steps', type=int, default=10,
                       help='Number of attack iterations')
    parser.add_argument('--attack_start', type=int, default=0,
                       help='Frame to start attacking')
    parser.add_argument('--attack_step', type=int, default=1,
                       help='Attack every N frames')
    parser.add_argument('--adv_weight_set', type=str, default='1',
                       choices=['1', '2', '3', '4'],
                       help='Weight set for adversarial loss')
    
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process multiple sequences if needed
    if ',' in args.seq_num:
        seq_nums = [args.seq_num]
    else:
        # Process all expressions for a video
        video_id = args.seq_num
        expr_dir = os.path.join(args.rmot_path, 'expression', video_id)
        if os.path.exists(expr_dir):
            expr_files = sorted(os.listdir(expr_dir))
            seq_nums = [f"{video_id},{expr_file}" for expr_file in expr_files if expr_file.endswith('.json')]
        else:
            seq_nums = [f"{video_id},1.json"]
    
    # Run attack on each sequence
    for seq_num in seq_nums:
        print(f"\n{'='*60}")
        print(f"Processing sequence: {seq_num}")
        print(f"{'='*60}")
        
        args.seq_num = seq_num
        detector = RMOTAdversarialDetector(args)
        
        detector.run(
            prob_threshold=0.7,
            area_threshold=100,
            ref_threshold=args.ref_threshold,
            vis=True
        )
        
        print(f"Completed sequence: {seq_num}\n")


if __name__ == '__main__':
    main()