# predicted as a batch
from params import par
from model import DeepVO
import numpy as np
from PIL import Image
import glob
import os
import time
import torch
from tqdm import tqdm
from data_helper import get_data_info, ImageSequenceDataset
from torch.utils.data import DataLoader
from helper import eulerAnglesToRotationMatrix
import torch.nn as nn
from torchvision import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.memory_summary(device=None, abbreviated=False)

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [-0.5, 0.5] range for each pixel
    perturbed_image = torch.clamp(perturbed_image, -0.5, 0.5)
    # Return the perturbed image
    return perturbed_image


# def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
#     """ Construct FGSM adversarial examples on the examples X"""

#     if randomize:
#         delta = torch.rand_like(X, requires_grad=True)
#         delta.data = delta.data * 2 * epsilon - epsilon
#     else:
#         delta = torch.zeros_like(X, requires_grad=True)
#     print(f'delta has shape {delta.shape}, X has shape {X.shape}, y has shape {y.shape}')
#     for t in range(num_iter):
#         loss = model.get_loss(X+delta, y)
#         loss.backward()
#         delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#         delta.grad.zero_()
#     return delta.detach()



def denorm(batch, mean=list(par.img_means), std = list(par.img_stds)):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
        
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
        
    # De-normalize using mean and std
    # We know that each "batch" has shape ([batch,seq,channel,width,height])
    
    batch_update = batch.clone()
    for channel in range(3):
        batch_update[:, :, channel, :, :] = batch[:, :, channel, :, :] * std[channel] + mean[channel]

    
    return batch_update


if __name__ == '__main__':    
    adversarial_attack = True
    non_uniform_attack= False
    # videos_to_test = ['04', '05', '07', '10', '09']
    videos_to_test = ['07']

    # Path
    load_model_path = par.load_model_path   #choose the model you want to load
    save_dir = 'result/'  # directory to save prediction answer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('using CUDA')
        M_deepvo = M_deepvo.cuda()
        M_deepvo.load_state_dict(torch.load(load_model_path))
    else:
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location={'cuda:0': 'cpu'}))
    print('Load model from: ', load_model_path)

    # Data
    n_workers = 1
    seq_len = int((par.seq_len[0]+par.seq_len[1])/2)
    overlap = seq_len - 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))
    batch_size = par.batch_size

    fd=open('test_dump.txt', 'w')
    fd.write('\n'+'='*50 + '\n')

    for test_video in (videos_to_test):
        df = get_data_info(folder_list=[test_video], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        df = df.loc[df.seq_len == seq_len]  # drop last
        dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
        df.to_csv('test_df.csv')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        #dataloader is now the test set
        #apply adversarial attack to this !
  
        # Predict
        M_deepvo.eval()
        M_deepvo.rnn.train()
        has_predict = False
        answer = [[0.0]*6, ]
        st_t = time.time()
        n_batch = len(dataloader)

        for i, batch in enumerate(dataloader):
            print('{} / {}'.format(i, n_batch), end='\r', flush=True)
            _, x, y = batch
            print(f"x has shape {x.shape}")
            # print(f'batch has length {len(batch)}')
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
  
            x.requires_grad = True

            if adversarial_attack:
                # print(f'x has shape{x.shape}, y has shape {y.shape}')
                output = M_deepvo.forward(x)
                y = y[:, 1:, :]
                angle_loss = torch.nn.functional.mse_loss(output[:,:,:3], y[:,:,:3])
                translation_loss = torch.nn.functional.mse_loss(output[:,:,3:], y[:,:,3:])
                
                # x.retain_grad()
                
                loss = (100 * angle_loss + translation_loss)

                M_deepvo.zero_grad()
                loss.backward()

                x_grad = x.grad.data
                # print(f'x_grad has shape {x_grad.shape}')
                x_denormed = denorm(x)
                print(x_denormed.max(),x_denormed.min())
                
                if non_uniform_attack:
                    if (i > 15) and (i < 100):
                        perturbed_image = fgsm_attack(x_denormed, 0.1, x_grad)
                        #Re-apply normalization:
                        perturbed_image_normalized = transforms.Normalize((list(par.img_means)), (list(par.img_stds)))(perturbed_image)
                        batch_predict_pose = M_deepvo.forward(perturbed_image_normalized)
                    else:
                        batch_predict_pose = M_deepvo.forward(x)
                        
                    
                else:
                    perturbed_image = fgsm_attack(x_denormed, 0.1, x_grad)
                    #Re-apply normalization:
                    perturbed_image_normalized = transforms.Normalize((list(par.img_means)), (list(par.img_stds)))(perturbed_image)
                    batch_predict_pose = M_deepvo.forward(perturbed_image_normalized)

                
            else:
                
                batch_predict_pose = M_deepvo.forward(x)


            # Record answer
            fd.write('Batch: {}\n'.format(i))
            for seq, predict_pose_seq in enumerate(batch_predict_pose):
                for pose_idx, pose in enumerate(predict_pose_seq):
                    fd.write(' {} {} {}\n'.format(seq, pose_idx, pose))


            batch_predict_pose = batch_predict_pose.data.cpu().numpy()
   
            if i == 0:
                for pose in batch_predict_pose[0]:
                    # use all predicted pose in the first prediction
                    for i in range(len(pose)):
                        # Convert predicted relative pose to absolute pose by adding last pose
                        pose[i] += answer[-1][i]
                    answer.append(pose.tolist())
                batch_predict_pose = batch_predict_pose[1:]

            # transform from relative to absolute 
            
            for predict_pose_seq in batch_predict_pose:
                # predict_pose_seq[1:] = predict_pose_seq[1:] + predict_pose_seq[0:-1]
                ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0]) #eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]])
                location = ang.dot(predict_pose_seq[-1][3:])
                predict_pose_seq[-1][3:] = location[:]

            # use only last predicted pose in the following prediction
                last_pose = predict_pose_seq[-1]
                for i in range(len(last_pose)):
                    last_pose[i] += answer[-1][i]
                # normalize angle to -Pi...Pi over y axis
                last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                answer.append(last_pose.tolist())

        print('len(answer): ', len(answer))
        print('expect len: ', len(glob.glob('{}{}/*.png'.format(par.image_dir, test_video))))
        print('Predict use {} sec'.format(time.time() - st_t))


        # Save answer
        if adversarial_attack:
            if non_uniform_attack:
                with open('{}/selective_out_adversarial_{}.txt'.format(save_dir, test_video), 'w') as f:
                    for pose in answer:
                        if type(pose) == list:
                            f.write(', '.join([str(p) for p in pose]))
                        else:
                            f.write(str(pose))
                        f.write('\n')
            else:
                with open('{}/out_adversarial_{}.txt'.format(save_dir, test_video), 'w') as f:
                    for pose in answer:
                        if type(pose) == list:
                            f.write(', '.join([str(p) for p in pose]))
                        else:
                            f.write(str(pose))
                        f.write('\n')
        else:
            
            with open('{}/out_{}.txt'.format(save_dir, test_video), 'w') as f:
                        for pose in answer:
                            if type(pose) == list:
                                f.write(', '.join([str(p) for p in pose]))
                            else:
                                f.write(str(pose))
                            f.write('\n')

        # Calculate loss
        gt_pose = np.load('{}{}.npy'.format(par.pose_dir, test_video))  # (n_images, 6)
        loss = 0
        for t in range(len(gt_pose)):
            angle_loss = np.sum((answer[t][:3] - gt_pose[t,:3]) ** 2)
            translation_loss = np.sum((answer[t][3:] - gt_pose[t,3:6]) ** 2)
            loss = (100 * angle_loss + translation_loss)
        loss /= len(gt_pose)
        print('Loss = ', loss)
        print('='*50)
