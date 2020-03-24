import torch as th
from s3dg import S3D

def main():
    # Instantiate the model
    net = S3D('s3d_dict.npy', 512)

    # Load the model weights
    net.load_state_dict(th.load('s3d_howto100m.pth'))

    # Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1]
    video = th.rand(2, 3, 32, 224, 224)

    # Evaluation mode
    net = net.eval()

    # Video inference
    video_output = net(video)

    # Text inference
    text_output = net.text_module(['open door', 'cut tomato'])

    print(text_output)



if __name__ == "__main__":
    main()