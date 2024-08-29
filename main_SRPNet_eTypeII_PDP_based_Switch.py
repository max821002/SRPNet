import argparse
from math import log10
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from scipy.fftpack import dct, idct, fft, ifft
from scipy.io import loadmat, savemat
import mat73
from model import SRCsiNet_BD, UL_ISTANet, SR_ISTANet9, SR_ISTANet10, SR_ISTANet11, SR_ISTANet12, SR_ISTANet4, SR_ISTANet5, SR_ISTANet6, SR_ISTANet7, SR_ISTANet8, SR_ISTANet, ISTANet, SIMaskNet, SIMaskNet2, CsiNetPro_Encoder, CsiNetPro_Decoder, CsiNetPro_Encoder_AF, CsiNetPro_Decoder_AF
from model import ULCSISwitch, ULCSISwitch_norm, ULCSISwitch_norm_2, ULCSISwitch_norm_multilayer, ULCSISwitch_norm_multilayer_2, DualNet_Encoder, DualNet_Decoder, DualNet_Decoder2

nb_epochs = 2000
device_no = 0
flag_norm = 1
flag_noisy = 1
flag_denoise = 1
flag_twostage = 0
loss_type = 0
start_epoch = 2000

us_set = [1]
model_usage_set = [0]
#loss_type = 0
channel_type = 1
BW_set = 0
side_info_density_set = [1]
max_ref_TTI = 2
No_RB = 96 # 32*12*30K = 11520K = 11.5M
CR_set = [1]
SC_spacing = 15
no_extra_pilot = 256
LayerNo = 10
No_RB_per_SB = 8 # D_RBpSB
D_ant_set = [1]
sc = SC_spacing
Qp = 4
Qa = 7

#Mv = 48
#R_set = [4]

Mv = 48
R_set = [4]

training_size_augment = 2
slope = 1
cost_itp = 1
cost_srcsinet = 1000
lambdaa_set = [0.00001,0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
#lambdaa_set = [0.00001,0.00005]
def numpy_sort_4d(arr, descending=False):
    """
    Sorts a 4-dimensional numpy array along the last dimension, mimicking the behavior of torch.sort().
    
    Parameters:
    - arr: A 4-dimensional numpy array to be sorted.
    - descending: Boolean, if True sorts the array in descending order, else in ascending order.
    
    Returns:
    - sorted_array: A 4-dimensional numpy array sorted in the specified order along the last dimension.
    - indices: The indices of the original array that give the sorted order along the last dimension.
    """
    # np.argsort returns the indices that would sort the array along the last dimension.
    indices = np.argsort(arr, axis=-1)
    
    if descending:
        # If sorting in descending order, reverse the indices along the last dimension.
        indices = np.flip(indices, axis=-1)
    
    # Generate a grid of indices for the first three dimensions to accompany the sorted indices for the last dimension.
    # np.ogrid generates an open mesh of indices which, when broadcasted together, can index the array except for the last dimension.
    ixgrid = np.ogrid[[slice(x) for x in arr.shape[:-1]]]
    # Add the sorted indices as the last part of the index.
    ixgrid.append(indices)
    
    # Use the indices to sort the array along the last dimension.
#    sorted_array = arr[tuple(ixgrid)]
    
    return indices

for idx_lambda in range(len(lambdaa_set)):
    R = R_set[0]
    lambdaa = lambdaa_set[idx_lambda]
    D_ant = D_ant_set[0]
    for idx_upsampling in range(len(us_set)):
      flag_learning_upsampling = us_set[idx_upsampling]
      filedir = 'Z:\\UCD\\data\\Sim_UMa_2GHz_'+str(SC_spacing)+'kHz_2M_MM\\'
#      Sim_UMa_2GHz_15kHz_2M_MM
#     
#      if idx_lambda == 0:
#          mat_dl = mat73.loadmat(filedir+'H_dl_Qp='+str(Qp)+'_Qa='+str(Qa)+'_'+str(sc)+'KHz_B16_etypeII_non_trunc.mat')
    
      h_dl_RB = mat_dl['H_RB_channel'][:,:,:No_RB]
      h_ul_RB = mat_dl['H_RB_UL_channel'][:,:,:No_RB]
      f_dl_RB = mat_dl['H_RB_precoder'][:,:,:No_RB]
      
      ## data preprocessing
      input_ul = np.reshape(h_ul_RB,[h_ul_RB.shape[0],1,h_ul_RB.shape[1],h_ul_RB.shape[2]])
      label_dl = np.reshape(f_dl_RB,[f_dl_RB.shape[0],1,f_dl_RB.shape[1],f_dl_RB.shape[2]])
      
      
      ref_idx = np.array(range(0,No_RB,No_RB_per_SB))
    
      reverse_idx = np.zeros_like(ref_idx)
      for idx in range(reverse_idx.size):
          reverse_idx[idx] = np.argwhere(ref_idx==ref_idx[idx])
      ## model
      for idx_model_run in range(len(model_usage_set)):
        idx_model = model_usage_set[idx_model_run]
        if idx_model == 0:
          model_name = 'SRCsiNet_BD'
          model_name2 = 'ULCSISwitch_norm_multilayer_2'
          BATCH_SIZE = 16
        cuda = 'store_true'
        device = torch.device(f"cuda:{device_no}" if (torch.cuda.is_available() and cuda ) else "cpu")
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        class data_set(Dataset):
          def __init__(self, input_data1, input_data2, label_data):
            self.input1 = input_data1
            self.input2 = input_data2
            self.label = label_data
            
          def __len__(self):
            return len(self.input1)
          def __getitem__(self, index):
            return self.input1[index], self.input2[index], self.label[index]
        
        input_dl_all = label_dl.copy()
                  
        input_dl_all = input_dl_all[:,:,:,0:No_RB:No_RB//Mv]
        input_dl_all = np.fft.ifft(input_dl_all, axis=-1)
        tmp = np.zeros_like(input_dl_all)
          # choose largest Mv/R delay taps
        indices = numpy_sort_4d(np.abs(input_dl_all),descending=True)
        for idx_sample in range(input_dl_all.shape[0]):
          for idx_l in range(input_dl_all.shape[2]):
               input_dl_all[idx_sample,0,idx_l,indices[idx_sample,0,idx_l, Mv//R:]] = np.zeros_like(input_dl_all[idx_sample,0,idx_l, indices[idx_sample,0,idx_l, Mv//R:]])
          
        input_dl_all = np.fft.fft(input_dl_all, axis=-1)

        
        
        
        input_ul_all = input_ul
        label_dl_all = label_dl
    
        TEST_SIZE = int(input_dl_all.shape[0]/10)
        VAL_SIZE = int(input_dl_all.shape[0]/10)
        TRAIN_SIZE = int(input_dl_all.shape[0]/10*8)
        
        si_test = input_ul_all[0:TEST_SIZE,]
        si_val = input_ul_all[TEST_SIZE:TEST_SIZE+VAL_SIZE,]
        si_train = input_ul_all[TEST_SIZE+VAL_SIZE:TEST_SIZE+VAL_SIZE+TRAIN_SIZE,]
        
        input_test = input_dl_all[0:TEST_SIZE,]
        input_val = input_dl_all[TEST_SIZE:TEST_SIZE+VAL_SIZE,]
        input_train = input_dl_all[TEST_SIZE+VAL_SIZE:TEST_SIZE+VAL_SIZE+TRAIN_SIZE,]
        
        label_test = label_dl_all[0:TEST_SIZE,]
        label_val = label_dl_all[TEST_SIZE:TEST_SIZE+VAL_SIZE,]
        label_train = label_dl_all[TEST_SIZE+VAL_SIZE:TEST_SIZE+VAL_SIZE+TRAIN_SIZE,]
        
        input_array_size = 128//D_ant
        output_array_size = 128
        tmp = np.zeros([TRAIN_SIZE,2,input_array_size,input_train.shape[-1]])
        tmp = np.concatenate((np.real(input_train),np.imag(input_train)),axis=-3)
        input_train = tmp.copy()
        
        tmp = np.zeros([VAL_SIZE,2,input_array_size,input_val.shape[-1]])
        tmp = np.concatenate((np.real(input_val),np.imag(input_val)),axis=-3)
        input_val = tmp.copy()
        
        tmp = np.zeros([TEST_SIZE,2,input_array_size,input_test.shape[-1]])
        tmp = np.concatenate((np.real(input_test),np.imag(input_test)),axis=-3)
        input_test = tmp.copy()
        
        tmp = np.zeros([TRAIN_SIZE,2,output_array_size,label_train.shape[-1]])
        tmp = np.concatenate((np.real(label_train),np.imag(label_train)),axis=-3)
        label_train = tmp.copy()
        
        tmp = np.zeros([VAL_SIZE,2,output_array_size,label_val.shape[-1]])
        tmp = np.concatenate((np.real(label_val),np.imag(label_val)),axis=-3)
        label_val = tmp.copy()
        
        tmp = np.zeros([TEST_SIZE,2,output_array_size,label_test.shape[-1]])
        tmp = np.concatenate((np.real(label_test),np.imag(label_test)),axis=-3)
        label_test = tmp.copy()
        
        tmp = np.zeros([TRAIN_SIZE,2,output_array_size,si_train.shape[-1]])
        tmp = np.concatenate((np.real(si_train),np.imag(si_train)),axis=-3)
        si_train = tmp.copy()
        
        tmp = np.zeros([VAL_SIZE,2,output_array_size,si_val.shape[-1]])
        tmp = np.concatenate((np.real(si_val),np.imag(si_val)),axis=-3)
        si_val = tmp.copy()
        
        tmp = np.zeros([TEST_SIZE,2,output_array_size,si_test.shape[-1]])
        tmp = np.concatenate((np.real(si_test),np.imag(si_test)),axis=-3)
        si_test = tmp.copy()
        
        for idx_norm in range(TRAIN_SIZE):
          norm_factor = np.max(np.abs(label_train[idx_norm,]))
          input_train[idx_norm,] = input_train[idx_norm,]/norm_factor
          label_train[idx_norm,] = label_train[idx_norm,]/norm_factor
          si_train[idx_norm,] = si_train[idx_norm,]/norm_factor
        for idx_norm in range(TEST_SIZE):
          norm_factor = np.max(np.abs(label_test[idx_norm,]))
          input_test[idx_norm,] = input_test[idx_norm,]/norm_factor
          label_test[idx_norm,] = label_test[idx_norm,]/norm_factor
          si_test[idx_norm,] = si_test[idx_norm,]/norm_factor
        for idx_norm in range(VAL_SIZE):
          norm_factor = np.max(np.abs(label_val[idx_norm,]))
          input_val[idx_norm,] = input_val[idx_norm,]/norm_factor
          label_val[idx_norm,] = label_val[idx_norm,]/norm_factor
          si_val[idx_norm,] = si_val[idx_norm,]/norm_factor
        
        
        
        dftmtx = np.fft.fft(np.eye(No_RB),axis=0).astype(np.complex64)
        for idx in range(dftmtx.shape[0]):
            dftmtx[:,idx] = dftmtx[:,idx]/np.linalg.norm(dftmtx[:,idx])
        dftmtx_ds = dftmtx[ref_idx,:]
        
        idftmtx = np.fft.ifft(np.eye(No_RB),axis=0).astype(np.complex64)
        for idx in range(idftmtx.shape[0]):
            idftmtx[:,idx] = idftmtx[:,idx]/np.linalg.norm(idftmtx[:,idx])
        idftmtx_ds = idftmtx[:,ref_idx]
        
        testset = data_set(torch.Tensor(input_test), torch.Tensor(si_test), torch.Tensor(label_test))
        trainset = data_set(torch.Tensor(input_train), torch.Tensor(si_train), torch.Tensor(label_train))
        valset = data_set(torch.Tensor(input_val), torch.Tensor(si_val), torch.Tensor(label_val))
        
        testloader = DataLoader(dataset=testset, batch_size = BATCH_SIZE, shuffle = True)
        trainloader = DataLoader(dataset=trainset, batch_size = BATCH_SIZE, shuffle = True)
        valloader = DataLoader(dataset=valset, batch_size = BATCH_SIZE, shuffle = True)
        
        for idx_cr in range(len(CR_set)):
            CR = CR_set[idx_cr]
#            Upsampler = SRCsiNet_BD([D_ant,(No_RB//Mv)]).to(device)
            Switch = ULCSISwitch_norm_multilayer_2().to(device)
#            criterion = nn.MSELoss()
            Upsampler = torch.load("model_"+model_name+"_20MHz_MM_practrical_Qp_Qa="+str(Qp)+"_"+str(Qa)+"_Mv="+str(Mv)+"_R="+str(R)+"_Dant"+str(D_ant)+"_"+str(sc)+'_B16_etypeII_KHz_pth').to(device)               
            for param in Upsampler.parameters():
                param.requires_grad = False
            def NMSEloss(output, target):
                output = output[:,0,:,:] + 1j*output[:,1,:,:]
                target = target[:,0,:,:] + 1j*target[:,1,:,:]
                target_norm = torch.linalg.matrix_norm(target)
                diff_norm = torch.linalg.matrix_norm(target-output)
                loss = torch.square(diff_norm/target_norm)
                return loss.sum()
            def CSloss(output, target):
                output = output[:,0,:,:] + 1j*output[:,1,:,:]
                target = target[:,0,:,:] + 1j*target[:,1,:,:]
                inner_product = torch.abs(torch.sum(torch.conj(output)*target,1))
                inner_product = inner_product.view(-1)
                target_norm = torch.linalg.matrix_norm(torch.reshape(torch.transpose(output,1,2),[output.size(0)*output.size(2),output.size(1),1]))
                output_norm = torch.linalg.matrix_norm(torch.reshape(torch.transpose(target,1,2),[output.size(0)*output.size(2),output.size(1),1]))
                loss = inner_product/target_norm/output_norm
                return loss.sum()/No_RB
            
#            criterion = nn.MSELoss()
            optimizer = optim.Adam(Switch.parameters(), lr = 0.001)
            
            training_loss = np.zeros(nb_epochs)
            test_loss = np.zeros(nb_epochs)
            val_loss = np.zeros(nb_epochs)
            
            best_loss = 0
            count = 0
            print(model_name)
            
            for epoch in range(nb_epochs):
              epoch_loss = 0
              epoch_loss_nmse = 0
              if epoch//100:
                  slope = slope*1
              
              for iteration, batch in enumerate(trainloader):
                input1 = batch[0].to(device) # 8xno_RB
                input2 = batch[1].to(device) # 8xno_RB*12
                target = batch[2].to(device) # 8xno_RB*12
                
                
                
                input = input1, input2
                
                optimizer.zero_grad()
                s = Switch(input2,slope)
                s = torch.reshape(s,[s.size(0),1,1,1])
                out_srcsinet, a, b, c, d = Upsampler(input,4//(No_RB//Mv))
                
                out_itp = nn.Upsample(scale_factor=tuple([1,No_RB//Mv]),mode='bilinear')(input1)
                
                out = s*out_srcsinet + (1-s)*out_itp
                
                
                loss = CSloss(out,target)
            
                cost = s*cost_srcsinet + (1-s)*cost_itp
                cost = torch.sum(cost)/s.size(0)
                total_loss = - loss + lambdaa*cost
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                loss_nmse = NMSEloss(out,target)
                epoch_loss_nmse += loss_nmse
                
              training_loss[epoch] = epoch_loss/TRAIN_SIZE
              print(f"Epoch {epoch}. Training NMSE: {10*torch.log10(epoch_loss_nmse/TRAIN_SIZE)}")
              print(f"Epoch {epoch}. Training loss: {epoch_loss/TRAIN_SIZE}")
              with torch.no_grad():
                epoch_loss = 0
                for iteration, batch in enumerate(testloader):
                  input1 = batch[0].to(device) # 8xno_RB
                  input2 = batch[1].to(device) # 8xno_RB*12
                  target = batch[2].to(device) # 8xno_RB*12
                  
                  
                    
                  input = input1, input2
                  s = Switch(input2,slope)
                  s = torch.reshape(s,[s.size(0),1,1,1])
                  out_srcsinet, a, b, c, d = Upsampler(input,4//No_RB_per_SB)
                  out_itp = nn.Upsample(scale_factor=tuple([1,No_RB//Mv]),mode='bilinear')(input1)
                  out = s*out_srcsinet + (1-s)*out_itp
    #              out = out.view([BATCH_SIZE,2,8,No_RB*12])
                  loss = CSloss(out,target)
            #          loss = criterion(out,target)
                  epoch_loss += loss
        #        test_loss[epoch] = 10*torch.log10(epoch_loss/TEST_SIZE)
                print(f"Epoch {epoch}. Test CS: {(epoch_loss/TEST_SIZE)}")
                epoch_loss = 0
                epoch_CS = 0
                for iteration, batch in enumerate(valloader):
                  input1 = batch[0].to(device) # 8xno_RB
                  input2 = batch[1].to(device) # 8xno_RB*12
                  target = batch[2].to(device) # 8xno_RB*12
                  
                    
                  input = input1, input2
                  s = Switch(input2,slope)
                  s = torch.reshape(s,[s.size(0),1,1,1])
                  out_srcsinet, a, b, c, d = Upsampler(input,4//No_RB_per_SB)
                  out_itp = nn.Upsample(scale_factor=tuple([1,No_RB//Mv]),mode='bilinear')(input1)
                  out = s*out_srcsinet + (1-s)*out_itp
    #              out = out.view([BATCH_SIZE,2,8,No_RB*12])
                  loss = CSloss(out,target)
                  epoch_CS += loss.item()
                  cost = s*cost_srcsinet + (1-s)*cost_itp
                  cost = torch.sum(cost)/s.size(0)
                  total_loss = - loss + lambdaa*cost
                  
                  epoch_loss += total_loss.item()
                val_loss[epoch] = epoch_loss/VAL_SIZE
                print(f"Epoch {epoch}. Val loss: {epoch_loss/VAL_SIZE}")
                print(f"Epoch {epoch}. Val CS: {epoch_CS/VAL_SIZE}")
                print(f"Epoch {epoch}. Val Cost: {(epoch_CS+epoch_loss)/VAL_SIZE}")
                if val_loss[epoch] < best_loss:
                  print("Saving...")
                  torch.save(Switch, "model_"+model_name2+"_20MHz_MM_practrical_Qp_Qa="+str(Qp)+"_"+str(Qa)+"_Mv="+str(Mv)+"_R="+str(R)+"_Dant"+str(D_ant)+"_"+str(sc)+"lambda="+str(lambdaa)+"_"+'_B16_etypeII_KHz_pth')               
                  count = 0
                else:
                  count += 1
                if epoch == 0:
                  best_loss = val_loss[0]
                else:
                  best_loss = np.min(val_loss[0:epoch])
              if count >= 100:
                break
        #    min_mse_idx = np.argmin(val_loss)
        #    min_mse = val_loss[min_mse_idx]
        #    mdic = {"min_mse": min_mse, "min_mse_idx": min_mse_idx}
        #    savemat("result_loss_" + model_name + "_side_info_density"+str(side_info_density) + '.mat',mdic)
            if flag_learning_upsampling == 1:
                del Upsampler
                torch.cuda.empty_cache()
            
            plt.plot(training_loss, label = "training loss")
            plt.plot(val_loss, label = "val loss")
        
        
        
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        