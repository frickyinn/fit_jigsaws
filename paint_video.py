import os
import numpy as np
import cv2
import torch
import torchvision
from tqdm import tqdm
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from estimator import PoseEstimator
from render import JIGSAWSRenderer


def colormap(rgb=True):
	color_list = np.array(
		[
			0.000, 0.000, 0.000,
			1.000, 1.000, 1.000,
			1.000, 0.498, 0.313,
			0.392, 0.581, 0.929,
			0.000, 0.447, 0.741,
			0.850, 0.325, 0.098,
			0.929, 0.694, 0.125,
			0.494, 0.184, 0.556,
			0.466, 0.674, 0.188,
			0.301, 0.745, 0.933,
			0.635, 0.078, 0.184,
			0.300, 0.300, 0.300,
			0.600, 0.600, 0.600,
			1.000, 0.000, 0.000,
			1.000, 0.500, 0.000,
			0.749, 0.749, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 1.000,
			0.667, 0.000, 1.000,
			0.333, 0.333, 0.000,
			0.333, 0.667, 0.000,
			0.333, 1.000, 0.000,
			0.667, 0.333, 0.000,
			0.667, 0.667, 0.000,
			0.667, 1.000, 0.000,
			1.000, 0.333, 0.000,
			1.000, 0.667, 0.000,
			1.000, 1.000, 0.000,
			0.000, 0.333, 0.500,
			0.000, 0.667, 0.500,
			0.000, 1.000, 0.500,
			0.333, 0.000, 0.500,
			0.333, 0.333, 0.500,
			0.333, 0.667, 0.500,
			0.333, 1.000, 0.500,
			0.667, 0.000, 0.500,
			0.667, 0.333, 0.500,
			0.667, 0.667, 0.500,
			0.667, 1.000, 0.500,
			1.000, 0.000, 0.500,
			1.000, 0.333, 0.500,
			1.000, 0.667, 0.500,
			1.000, 1.000, 0.500,
			0.000, 0.333, 1.000,
			0.000, 0.667, 1.000,
			0.000, 1.000, 1.000,
			0.333, 0.000, 1.000,
			0.333, 0.333, 1.000,
			0.333, 0.667, 1.000,
			0.333, 1.000, 1.000,
			0.667, 0.000, 1.000,
			0.667, 0.333, 1.000,
			0.667, 0.667, 1.000,
			0.667, 1.000, 1.000,
			1.000, 0.000, 1.000,
			1.000, 0.333, 1.000,
			1.000, 0.667, 1.000,
			0.167, 0.000, 0.000,
			0.333, 0.000, 0.000,
			0.500, 0.000, 0.000,
			0.667, 0.000, 0.000,
			0.833, 0.000, 0.000,
			1.000, 0.000, 0.000,
			0.000, 0.167, 0.000,
			0.000, 0.333, 0.000,
			0.000, 0.500, 0.000,
			0.000, 0.667, 0.000,
			0.000, 0.833, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 0.167,
			0.000, 0.000, 0.333,
			0.000, 0.000, 0.500,
			0.000, 0.000, 0.667,
			0.000, 0.000, 0.833,
			0.000, 0.000, 1.000,
			0.143, 0.143, 0.143,
			0.286, 0.286, 0.286,
			0.429, 0.429, 0.429,
			0.571, 0.571, 0.571,
			0.714, 0.714, 0.714,
			0.857, 0.857, 0.857
		]
	).astype(np.float32)
	color_list = color_list.reshape((-1, 3)) * 255
	if not rgb:
		color_list = color_list[:, ::-1]
	return color_list


color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def vis_add_mask(image, mask, color, alpha):
	color = np.array(color_list[color])
	mask = mask > 0.5
	image[mask] = image[mask] * (1-alpha) + color * alpha
	return image.astype('uint8')


def mask_painter(input_image, input_mask, mask_color=5, mask_alpha=0.7, contour_color=1, contour_width=3):
	assert input_image.shape[:2] == input_mask.shape, 'different shape between image and mask'
	# 0: background, 1: foreground
	mask = np.clip(input_mask, 0, 1)
	contour_radius = (contour_width - 1) // 2

	dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
	dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
	dist_map = dist_transform_fore - dist_transform_back
	# ...:::!!!:::...
	contour_radius += 2
	contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
	contour_mask = contour_mask / np.max(contour_mask)
	contour_mask[contour_mask>0.5] = 1.

	# paint mask
	painted_image = vis_add_mask(input_image.copy(), mask.copy(), mask_color, mask_alpha)
	# paint contour
	painted_image = vis_add_mask(painted_image.copy(), 1-contour_mask, contour_color, 1)

	return painted_image


def paint_video(video_path, output_path, backbone, checkpoint, obj_path, device):
    model = PoseEstimator(backbone, seg_ckpt=None)
    model.load_state_dict(torch.load(checkpoint)['model'])
    model = model.to(device)
    model.eval()
    jrenderer = JIGSAWSRenderer(obj_path, device=device)

    preprocess_fn = get_preprocessing_fn(backbone, 'imagenet')

    frames = []
    painted_frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
        
    avg = []
    for frame in tqdm(frames):
        image = frame
        frame = preprocess_fn(frame)
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.tensor(frame).float()
        x = model.predict_RT(frame.to(device)[None, ...])
        avg.append(x.cpu())
        rendered = jrenderer.render_batch_masks(x).cpu().numpy()

        painted_frame = image
        for i, r in enumerate(rendered[0, 1:]):
            if (r>0).sum() == 0:
                continue
            painted_frame = mask_painter(painted_frame, (r>0).astype('uint8'), mask_color=i+1)
        painted_frames.append(painted_frame)

    avg = torch.cat(avg)
    left, right = avg[:, :12].reshape(-1, 3, 4), avg[:, 12:].reshape(-1, 3, 4)
    print(left.mean(0))
    print(left.std(0))
    print(right.mean(0))
    print(right.std(0))
    generate_video_from_frames(painted_frames, os.path.join(output_path, os.path.split(video_path)[-1]), fps)
        

def generate_video_from_frames(frames, output_path, fps=30):
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


if __name__ == "__main__":
    video_name = 'Knot_Tying_I005_capture2.avi'
    video_path = os.path.join('../videos', video_name)
    output_path = os.path.join('../videos/painted')
    checkpoint = 'checkpoints/DR_2layer_20230924_0928/DR_2layer_model_best.pth'
    paint_video(video_path=video_path, output_path=output_path, backbone='resnet50', checkpoint=checkpoint, obj_path='GasCylinder02.obj', device='cuda:1')