import torch as th
import numpy as np
from s3dg import S3D
import numpy as np

def main():
    # see model input data
    # data = np.load('s3d_dict.npy')

    # Instantiate the model
    net = S3D('s3d_dict.npy', 512)

    # Load the model weights
    net.load_state_dict(th.load('s3d_howto100m.pth'))

    # Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1]
    # video1 = th.rand(2, 3, 32, 224, 224)
    # print(video1.shape)
    # print(type(video1))
    video = th.from_numpy(np.load("../video_feature_extractor/output/_0flfBHjVKU_features.npy"))
    print(video.shape)
    print(type(video))

    # Evaluation mode
    net = net.eval()

    # Video inference
    '''
    video_output is a dictionary containing two keys:

        video_embedding: This is the video embedding (size 512) from the joint text-video space. 
                        It should be used to compute similarity scores with text inputs using the text embedding.
        
        mixed_5c: This is the global averaged pooled feature from S3D of dimension 1024. 
                This should be use for classification on downstream tasks.
    '''
    video_output = net(video)
    print(video_output['mixed_5c'])
    print(video_output['mixed_5c'].shape)
    print(type(video_output['mixed_5c']))
    #
    # # Text inference
    # text_output = net.text_module(['open door', 'cut tomato'])

    # print(text_output)



if __name__ == "__main__":
    main()