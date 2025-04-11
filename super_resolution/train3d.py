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
from VidIQ.autoencoder.util import WORLD_SIZE, ddp_setup, print_device_info
from VidIQ.super_resolution.models import MODEL_NAME, SCALE_RATIO, get_models

# Define hyperparameter
sampler_rate = 0.2
num_epochs = 20
save_epoch = 20
loss_index = 2
# Learning rate is positively correlated with batch size
batch_size = SCALE_RATIO-1
learning_rate = WORLD_SIZE * 5e-4


def main(rank, train_dataset):
    ddp_setup(rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, sampler=DistributedSampler(train_dataset))
    if rank == 0:
        print(
            f'Length of training set (i.e. number of batch) is {len(train_loader)}')

    # Define model and optimizer
    # model3d, model3d_path = get_models(
    #     is_load=False, rank=rank, network3d=True)
    model3d, model3d_path = get_models(
        is_load=True, rank=rank, network3d=True, transfer="2")
    optimizer = optim.Adam(model3d.parameters(), lr=learning_rate)
    # Multiply learning rate by 0.5 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # Define loss function
    criterion = loss_function(loss_index, rank)

    # Training model
    rev = ReshapeVideo()
    for epoch in range(num_epochs):
        train_loss = 0
        batch_index = 1
        train_loader.sampler.set_epoch(epoch)
        for i, (videos, lr_videos, paths) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward propagation
            videos = videos.to(rank)
            lr_videos = lr_videos.to(rank)
            outputs = model3d(lr_videos)

            # Calculate loss (and accuracy)
            output_frames = rev(outputs)
            video_frames = rev(videos)
            loss = criterion(output_frames, video_frames)
            dist.all_reduce(loss)
            loss /= WORLD_SIZE

            # Error propagation and optimization
            loss.backward()
            optimizer.step()
            del videos, lr_videos
            torch.cuda.empty_cache()
            train_loss += loss.item()
            if rank == 0:
                print(
                    f"Epoch [{epoch+1}] batch [{batch_index}/{len(train_loader)}]", end="\r")
            batch_index += 1

        # Automatically adjust learning rate
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        if rank == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], LR: {lr}, Loss: {train_loss/len(train_loader):.4f}')

        if (epoch+1) % save_epoch == 0:
            if rank == 0:
                print(f"Epoch [{epoch+1}], saving trained weights")
                torch.save(model3d.module.state_dict(), model3d_path)

    if rank == 0:
        torch.save(model3d.module.state_dict(), model3d_path)
    destroy_process_group()


if __name__ == "__main__":
    print_device_info()
    print(f'SR model is {MODEL_NAME}x{SCALE_RATIO}')
    # Prepare training dataset
    train_dataset = VideoDataset(
        VIDEO_DIR, CNN_FRAME_NUM, sampler_rate, reduce_step=2*SCALE_RATIO if SCALE_RATIO >= 3 else 2, scale_ratio=SCALE_RATIO)
    print(f'Training set contains {len(train_dataset)} videos')
    print(f'Batch size is {batch_size*WORLD_SIZE}')
    mp.spawn(main, args=(train_dataset,), nprocs=WORLD_SIZE)
