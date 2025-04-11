import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from VidIQ.autoencoder.dataset import IMAGE_DIR, ImageDataset
from VidIQ.autoencoder.loss import loss_function
from VidIQ.autoencoder.network2d import get_networks, print_autoencoder_info
from VidIQ.autoencoder.util import WORLD_SIZE, ddp_setup, print_device_info
from VidIQ.super_resolution.models import (LR_IMAGE_DIR, MODEL_NAME,
                                            SCALE_RATIO, get_models)

# Define hyperparameter
CODEC_MODE = "encoder"
sampler_rate = 0.5
num_epochs = 20
save_epoch = 20
loss_index = 4
'''
Small initial learning rate may result in local minima and colorful mosaic images
When training loss 3-6, weights of loss 2 should be loaded as starting point
'''
# Learning rate is positively correlated with batch size
if loss_index < 3:
    batch_size = SCALE_RATIO+1
    learning_rate = WORLD_SIZE * 1e-3
else:
    batch_size = SCALE_RATIO
    learning_rate = WORLD_SIZE * 1e-4


def main(rank, train_dataset):
    ddp_setup(rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, sampler=DistributedSampler(train_dataset))
    if rank == 0:
        print(
            f'Length of training set (i.e. number of batch) is {len(train_loader)}')

    # Define model and optimizer
    if CODEC_MODE == "encoder":
        encoder, encoder_path, decoder, decoder_path = get_networks(
            is_load=True, rank=rank, joint=True)
        model, _ = get_models(is_load=True, rank=rank)
        params = [{'params': net.parameters()} for net in [encoder, decoder]]
    elif CODEC_MODE == "decoder":
        encoder, encoder_path = get_networks(
            is_load=True, rank=rank, is_decoder=False, joint=True)
        # model, model_path = get_models(
        #     is_load=True, rank=rank, integrated=True, transfer="2.4.8")
        model, model_path = get_models(
            is_load=False, rank=rank, integrated=True)
        '''
        Too deep convolutional networks may result in gradient disappearance
          and gray images, thus only SR model parameters are added to optimizer
        '''
        params = [{'params': net.parameters()} for net in [model]]

    optimizer = optim.Adam(params, lr=learning_rate)
    if CODEC_MODE == "encoder":
        # Multiply learning rate by 0.1 every 15 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    elif CODEC_MODE == "decoder":
        # Multiply learning rate by 0.1 every 5 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # Define loss function
    criterion = loss_function(loss_index, rank)

    # Training model
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        batch_index = 1
        train_loader.sampler.set_epoch(epoch)
        for i, (imgs, lr_imgs, paths) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward propagation
            imgs = imgs.to(rank)
            lr_imgs = lr_imgs.to(rank)
            codes = encoder(lr_imgs)
            if CODEC_MODE == "encoder":
                outputs = decoder(codes)
                outputs = model(outputs)
            elif CODEC_MODE == "decoder":
                outputs = model(codes)

            # Calculate loss (and accuracy)
            if loss_index < 3:
                loss = criterion(outputs, imgs)
            else:
                loss, acc = criterion(outputs, imgs)
                acc = torch.tensor(acc).to(rank)
                dist.all_reduce(acc)
                acc /= WORLD_SIZE
            dist.all_reduce(loss)
            loss /= WORLD_SIZE

            # Error propagation and optimization
            loss.backward()
            optimizer.step()
            del imgs, lr_imgs
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
                torch.save(encoder.module.state_dict(), encoder_path)
                if CODEC_MODE == "encoder":
                    torch.save(decoder.module.state_dict(), decoder_path)
                elif CODEC_MODE == "decoder":
                    torch.save(model.module.state_dict(), model_path)

    if rank == 0:
        torch.save(encoder.module.state_dict(), encoder_path)
        if CODEC_MODE == "encoder":
            torch.save(decoder.module.state_dict(), decoder_path)
        elif CODEC_MODE == "decoder":
            torch.save(model.module.state_dict(), model_path)
    destroy_process_group()


if __name__ == "__main__":
    print_device_info()
    print_autoencoder_info()
    print(f'SR model is {MODEL_NAME}x{SCALE_RATIO}')
    # Prepare training dataset
    train_dataset = ImageDataset(
        IMAGE_DIR, sampler_rate, reduce_step=4 if loss_index == 6 else 0, lr_dir=LR_IMAGE_DIR)
    print(f'Training set contains {len(train_dataset)} images')
    print(f'Batch size is {batch_size*WORLD_SIZE}')
    mp.spawn(main, args=(train_dataset,), nprocs=WORLD_SIZE)
