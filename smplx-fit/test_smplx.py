import smplx
import torch


if __name__=='__main__':

    model_path = './models/smplx/SMPLX_MALE.npz'

    dtype = torch.float32
    model_params = dict(model_path=model_path,
                        model_type='smplx',
                        use_pca=False,
                        dtype=dtype,
                        )
    
    model = smplx.create(gender='male', **model_params)

    device = torch.device('cuda')
    model.to(device)
    pose = torch.rand(1,63).to(device)
    output = model(body_pose=pose)
