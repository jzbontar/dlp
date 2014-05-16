require 'image'                                                                                 
require 'torch'                                                                                 
require 'cunn'                                                                                  
require 'cutorch'                                                                               
require 'nn'                                                                                    
require 'prof-torch'                                                                            
prof.clear()                                                                                    
                                                                                                
cutorch.setDevice(1)                                                                            
                                                                                                
prof.tic('load')                                                                                
--X,Y = torch_datasets:cifar(3)                                                                 
X = torch.FloatTensor(torch.FloatStorage('/home/tom/datasets/cifar_whitened/X')):resize(60000, 3, 32, 32):cuda()
collectgarbage()
prof.toc('load')                                                                                               
                                                                                                               
lamvals = {1.77827941e-05,   3.16227766e-05, 5.62341325e-05,   1.00000000e-04,   1.77827941e-04, 3.16227766e-04,   5.62341325e-04 }                                                                                           
--for lami = 1,#lamvals do                                                                                       
lami = 3
lambda_l1 = lamvals[lami]                                                                                      

enc = torch.load(string.format('enc_%.7f.net',lambda_l1))      
dec = torch.load(string.format('dec_%.7f.net',lambda_l1))      

ims = {}
n=16

function zoom(im, f)
    local cp = torch.Tensor(im:size(1),im:size(2)*f, im:size(3)*f)
    for c=1,im:size(1) do -- 1 or 3 channeld
        for i=1,im:size(2) do
            for j=1,im:size(3) do
                cp[{c,{(i-1)*f+1, i*f}, {(j-1)*f+1, j*f}}] = im[{c,i,j}]
            end
        end
    end
    return cp
end

for i=1,n do
    ims[#ims+1] = zoom(enc:get(1).weight[{i}],10)
end
for i=1,n do
    wt = dec:get(1).weight:transpose(1,2)
    ims[#ims+1] = zoom(wt[{i}],10)
end
print('average activation')
avac = torch.Tensor(32,28,28)
bs = 2048
for t = 1,X:size(1)-bs, bs do
    Xb= X:narrow(1, t, bs)
    Z=enc:forward(Xb)
    acs=torch.gt(Z:double(),0):sum(1):squeeze()
    avac:add(acs:double())
end
avac:div(t+bs-1)
for i=1,n do
    padded=torch.zeros(1,32,32)
    --padded[{1,{3,30},{3,30}}] = avac[{i}]
    padded[{{3,30},{3,30}}] = avac[{i}]
    ims[#ims+1] = zoom(padded,10)
end

im = image.toDisplayTensor(ims,3,n)
image.save(string.format('filters_lam_%.6f.png', lambda_l1),im)

--end
