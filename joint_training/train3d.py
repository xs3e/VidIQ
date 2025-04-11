import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from VidIQ.autoencoder.dataset import (CNN_FRAME_NUM, VIDEO_DIR, ReshapeVideo,
                                        VideoDataset)
from VidIQ.autoencoder.loss import loss_function
from VidIQ.autoencoder.network3d import (get_networks3d,
                                          print_autoencoder3d_info)
from VidIQ.autoencoder.util import WORLD_SIZE, ddp_setup, print_device_info
from VidIQ.super_resolution.models import MODEL_NAME, SCALE_RATIO, get_models

# Define hyperparameter
CODEC_MODE = "encoder"
sampler_rate = 0.3
num_epochs = 40
save_epoch = 20
loss_index = 2
'''
Small initial learning rate may result in local minima and colorful mosaic images
When training loss 3-6, weights of loss 2 should be loaded as starting point
'''
# Learning rate is positively correlated with batch size
if loss_index < 3:
    batch_size = SCALE_RATIO
    learning_rate = WORLD_SIZE * 5e-4
else:
    batch_size = SCALE_RATIO-1
    learning_rate = WORLD_SIZE * 5e-5


def main(rank, train_dataset):
    ddp_setup(rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, sampler=DistributedSampler(train_dataset))
    if rank == 0:
        print(
            f'Length of training set (i.e. number of batch) is {len(train_loader)}')

    # Define model and optimizer
    if CODEC_MODE == "encoder":
        encoder3d, encoder3d_path, decoder3d, decoder3d_path = \
            get_networks3d(is_load=False, rank=rank, joint=True)
        model3d, _ = get_models(
            is_load=True, rank=rank, network3d=True)
        params = [{'params': net.parameters()}
                  for net in [encoder3d, decoder3d]]
    elif CODEC_MODE == "decoder":
        encoder3d, encoder3d_path = get_networks3d(
            is_load=True, rank=rank, is_decoder=False, joint=True)
        # model3d, model3d_path = get_models(
        #     is_load=True, rank=rank, integrated=True, network3d=True, transfer='2.4.16.3d')
        model3d, model3d_path = \
            get_models(is_load=False, rank=rank,
                       integrated=True, network3d=True)
        '''
        Too deep convolutional networks may result in gradient disappearance
          and gray images, thus only SR model parameters are added to optimizer
        '''
        params = [{'params': net.parameters()}for net in [model3d]]

    optimizer = optim.Adam(params, lr=learning_rate)
    # Multiply learning rate by 0.1 every 10 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Define loss function
    criterion = loss_function(loss_index, rank)

    # Training model
    rev = ReshapeVideo()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        batch_index = 1
        train_loader.sampler.set_epoch(epoch)
        for i, (videos, lr_videos, paths) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward propagation
            videos = videos.to(rank)
            lr_videos = lr_videos.to(rank)
            if CODEC_MODE == "encoder":
                codes = encoder3d(lr_videos)
                outputs = decoder3d(codes)
                outputs = model3d(outputs)
            elif CODEC_MODE == "decoder":
                codes = encoder3d(lr_videos)
                outputs = model3d(codes)

            # Calculate loss (and accuracy)
            output_frames = rev(outputs)
            video_frames = rev(videos)
            if loss_index < 3:
                loss = criterion(output_frames, video_frames)
            else:
                loss, acc = criterion(output_frames, video_frames)
                acc = torch.tensor(acc).to(rank)
                dist.all_reduce(acc)
                acc /= WORLD_SIZE
            dist.all_reduce(loss)
            loss /= WORLD_SIZE

            # Error propagation and optimization
            loss.backward()
            optimizer.step()
            del videos, lr_videos
            torch.cuda.empty_cache()
            train_loss += loss.item()
            if loss_index >= 3:
                train_acc += acc.item()
            if rank == 0:
                print(
                    f"Epoch [{epoch+1}] batch [{batch_index}/{len(train_loader)}]", end="\r")
            batch_index += 1

        # Automatically adjust learning rate
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        if rank == 0:
            if loss_index < 3:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], LR: {lr}, Loss: {train_loss/len(train_loader):.4f}')
            else:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], LR: {lr}, Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc/len(train_loader):.4f}')

        if (epoch+1) % save_epoch == 0:
            if rank == 0:
                print(f"Epoch [{epoch+1}], saving trained weights")
                torch.save(encoder3d.module.state_dict(), encoder3d_path)
                if CODEC_MODE == "encoder":
                    torch.save(decoder3d.module.state_dict(), decoder3d_path)
                elif CODEC_MODE == "decoder":
                    torch.save(model3d.module.state_dict(), model3d_path)

    if rank == 0:
        torch.save(encoder3d.module.state_dict(), encoder3d_path)
        if CODEC_MODE == "encoder":
            torch.save(decoder3d.module.state_dict(), decoder3d_path)
        elif CODEC_MODE == "decoder":
            torch.save(model3d.module.state_dict(), model3d_path)
    destroy_process_group()


if __name__ == "__main__":
    print_device_info()
    print_autoencoder3d_info()
    print(f'SR model is {MODEL_NAME}x{SCALE_RATIO}')
    # Prepare training dataset
    train_dataset = VideoDataset(
        VIDEO_DIR, CNN_FRAME_NUM, sampler_rate,
        reduce_step=8 if loss_index == 6 else 6, scale_ratio=SCALE_RATIO)
    print(f'Training set contains {len(train_dataset)} videos')
    print(f'Batch size is {batch_size*WORLD_SIZE}')
    mp.spawn(main, args=(train_dataset,), nprocs=WORLD_SIZE)
