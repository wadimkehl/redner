import pyredner
import numpy as np
import torch
import cv2

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_use_gpu(False)

cam = pyredner.Camera(
    position=torch.tensor([0.0, 0.0, 0]),
    look_at=torch.tensor([0.0, 0.0, 1.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    fov=torch.tensor([35.0]),  # in degree
    fx=torch.tensor([0.8]),
    fy=torch.tensor([0.8]),
    ox=torch.tensor([0.5]),
    oy=torch.tensor([0.5]),
    clip_near=1e-2,  # needs to > 0
    resolution=(500, 500),
    fisheye=False,
    pinhole=True)

mat = pyredner.Material(torch.tensor([0.0, 0, 0.5], device=pyredner.get_device()))

vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
vertices[:, 2] += 3.

sphere = pyredner.Shape(vertices=vertices, indices=indices, uvs=uvs, normals=normals, material_id=0)
envmap = pyredner.EnvironmentMap(torch.ones([32, 64, 3], device=pyredner.get_device()))
scene_args = pyredner.RenderFunction.serialize_scene(
    scene=pyredner.Scene(cam, [sphere], [mat], [], envmap), num_samples=128, max_bounces=1)

target = pyredner.RenderFunction.apply(0, *scene_args)

cv2.imshow('target', target.clamp(0.0, 1.0).pow(1.0 / 2.2).detach().cpu().numpy())
cv2.waitKey()

fx = torch.tensor([0.7], device=pyredner.get_device(), requires_grad=True)
fy = torch.tensor([0.7], device=pyredner.get_device(), requires_grad=True)
ox = torch.tensor([0.5], device=pyredner.get_device(), requires_grad=True)
oy = torch.tensor([0.2], device=pyredner.get_device(), requires_grad=True)

# cam.fx = fx
# cam.fy = fy

optimizer = torch.optim.Adam([ox, oy], lr=0.00000)

for t in range(1000):

    optimizer.zero_grad()

    cam.ox = ox
    cam.oy = oy

    # cam = pyredner.Camera(
    #     position=torch.tensor([0.0, 0.0, 0]),
    #     look_at=torch.tensor([0.0, 0.0, 1.0]),
    #     up=torch.tensor([0.0, 1.0, 0.0]),
    #     fov=torch.tensor([35.0]),  # in degree
    #     fx=fx,
    #     fy=fy,
    #     ox=torch.tensor([0.5]),
    #     oy=torch.tensor([0.5]),
    #     clip_near=1e-2,  # needs to > 0
    #     resolution=(500, 500),
    #     fisheye=False,
    #     pinhole=True)

    scene_args = pyredner.RenderFunction.serialize_scene(
        scene=pyredner.Scene(cam, [sphere], [mat], [], envmap), num_samples=1, max_bounces=1)
    img = pyredner.RenderFunction.apply(t + 1, *scene_args)

    loss = (img - target).abs().mean()
    print('loss:', loss.item())
    print('iteration:', t)

    print('fx, fy, ox, oy:', fx.item(), fy.item(), ox.item(), oy.item())
    cv2.imshow('current', img.clamp(0.0, 1.0).pow(1.0 / 2.2).detach().cpu().numpy())
    cv2.waitKey()

    loss.backward()

    optimizer.step()

    #exit(0)
