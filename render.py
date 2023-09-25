import numpy as np
import torch

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader
)

def perpendicular_vector(x, y, w):
    b = torch.sqrt(torch.square(w) / (torch.square(x / y) + 1))
    a = -y / x * b
    return a, b

class JIGSAWSRenderer():
    def __init__(self, obj_path, dist=3, device='cuda', image_size=(480, 640)):
        self.verts, self.faces, _ = load_obj(obj_path, device=device)
        self.verts /= self.verts.max()
        self.verts -= self.verts.mean(0)
        self.verts *= 2
        self.verts[:, 1] *= 1.5
        self.oheight = self.verts[6163, 1] - self.verts[4170, 1]
        self.owidth = self.verts[:, 0].max()

        R, T = look_at_view_transform(dist=dist, elev=0, azim=0)
        self.camera = FoVPerspectiveCameras(device=device, R=R, T=T)
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-5,
            faces_per_pixel=50,
            bin_size=0
        )
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=self.raster_settings
            ),
            shader=SoftSilhouetteShader()
        )

        self.device = device

    # def render_one_hand(self, out):
    #     width = torch.sqrt(torch.sum(torch.square(out[:, 3:6]), 1)) * self.owidth
    #     v_top2bottom = out[:, 3:6] * self.oheight

    #     x1, y1 = perpendicular_vector(v_top2bottom[:, 0], v_top2bottom[:, 1], width)
    #     v1 = torch.vstack([x1, y1, torch.zeros_like(x1)]).T
    #     x2, d2 = perpendicular_vector(v_top2bottom[:, 0], v_top2bottom[:, 2], width)
    #     v2 = torch.vstack([x2, torch.zeros_like(x2), d2]).T

    #     Y = torch.stack([out[:, :3], out[:, :3] + v_top2bottom, out[:, :3] + v_top2bottom + v1, out[:, :3] + v_top2bottom + v2], dim=1)
    #     Y = torch.cat([Y, torch.ones_like(Y[..., :1])], dim=2)
    #     Y = Y.permute(0, 2, 1)

    #     o1 = self.verts[6163]
    #     o2 = self.verts[4170]
    #     X = torch.vstack([o1, o2, o2, o2])
    #     X[2, 0] += self.owidth
    #     X[3, 2] += self.owidth
    #     X = torch.vstack([X.T, torch.tensor([1., 1., 1., 1.]).to(self.device)])

    #     R = torch.matmul(Y.double(), torch.inverse(X.double()))
    #     translated = torch.hstack([self.verts, torch.ones(self.verts.shape[0], 1).to(self.device)])
    #     translated = torch.matmul(R, translated.T.double())
    #     translated = translated.permute(0, 2, 1)[..., :3].to(self.device).float()

    #     meshes = Meshes(verts=translated, faces=torch.stack([self.faces.verts_idx for _ in range(out.shape[0])]).to(self.device))
    #     silhouette_images = self.renderer_silhouette(meshes, cameras=self.camera)

    #     return silhouette_images[..., 3]

    def render_one_hand(self, out):
        out = out.reshape(-1, 4, 3)
        aug = torch.zeros_like(out[..., :1])
        aug[:, 3, :] = 1
        R = torch.cat([out, aug], dim=2).permute(0, 2, 1)
        
        translated = torch.hstack([self.verts, torch.ones(self.verts.shape[0], 1).to(self.device)])
        translated = torch.matmul(R, translated.T)
        translated = translated.permute(0, 2, 1)[..., :3].to(self.device)

        meshes = Meshes(verts=translated, faces=torch.stack([self.faces.verts_idx for _ in range(out.shape[0])]).to(self.device))
        silhouette_images = self.renderer_silhouette(meshes, cameras=self.camera)

        return silhouette_images[..., 3]

    def render_batch_masks(self, out):
        left_masks = self.render_one_hand(out[:, :12])
        right_masks = self.render_one_hand(out[:, 12:])
        background = 1 - torch.clip(left_masks + right_masks, min=0, max=1)
        masks = torch.stack([background, left_masks, right_masks], dim=1)

        return masks
    