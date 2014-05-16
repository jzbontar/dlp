require 'image'                                                                                 
require 'torch'                                                                                 
require 'cunn'                                                                                  
require 'cutorch'                                                                               
require 'nn'                                                                                    
--require 'jzt'                                                                                 
require 'L1CostMan'                                                                             
require 'prof-torch'                                                                            
prof.clear()                                                                                    
                                                                                                
cutorch.setDevice(1)                                                                            
                                                                                                
prof.tic('load')                                                                                
--X,Y = torch_datasets:cifar(3)                                                                 
X = torch.FloatTensor(torch.FloatStorage('/home/tom/datasets/cifar_whitened/X')):resize(60000, 3, 32, 32):cuda()
prof.toc('load')                                                                                               
                                                                                                               
lamvals = {1.77827941e-05,   3.16227766e-05, 5.62341325e-05,   1.00000000e-04,   1.77827941e-04, 3.16227766e-04,   5.62341325e-04 }                                                                                           
--for lami = 1,#lamvals do                                                                                       
lami = 3
lambda_l1 = lamvals[lami]                                                                                      

enc = torch.load(string.format('enc_%.7f.net',lambda_l1))      
dec = torch.load(string.format('dec_%.7f.net',lambda_l1))      



--end
