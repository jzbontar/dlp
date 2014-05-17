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
for lami = 1,#lamvals do                                                                                       
--lami = 3
lambda_l1 = lamvals[lami]           
print(' -- lambda = ' .. lambda_l1 .. ' -- ')

enc = torch.load(string.format('enc_%.7f.net',lambda_l1))      
dec = torch.load(string.format('dec_%.7f.net',lambda_l1))      

ims = {}
n=16
rows={}

function zoom(im, f)
    if im:nDimension()==3 then
        cp = torch.Tensor(im:size(1),im:size(2)*f, im:size(3)*f)
        for c=1,im:size(1) do -- 1 or 3 channeld
            for i=1,im:size(2) do
                for j=1,im:size(3) do
                    cp[{c,{(i-1)*f+1, i*f}, {(j-1)*f+1, j*f}}] = im[{c,i,j}]
                end
            end
        end
    else
        cp = torch.Tensor(im:size(1)*f,im:size(2)*f)
        for i=1,im:size(1) do
            for j=1,im:size(2) do
                cp[{{(i-1)*f+1, i*f}, {(j-1)*f+1, j*f}}] = im[{i,j}]
            end
        end
    end
    return cp
end

for i=1,n do
    ims[#ims+1] = zoom(enc:get(1).weight[{i}],12)
end
rows[#rows+1] = image.toDisplayTensor(ims,3,n)
ims={}
for i=1,n do
    wt = dec:get(1).weight:transpose(1,2)
    ims[#ims+1] = zoom(wt[{i}],12)
end
rows[#rows+1] = image.toDisplayTensor(ims,3,n)
ims={}
print('average activation')
avac = torch.CudaTensor(32,28,28)
bs = 2048
N=0
Zac = torch.CudaTensor(bs,32,28,28)
--for t = 1,X:size(1)-bs, bs do
for t = 1,50000-bs, bs do
    Xb= X:narrow(1, t, bs)
    Z=enc:forward(Xb)
    Zac:gt(Z,0)
    acs=Zac:sum(1)[1]
    avac:add(acs)
    N=t+bs-1
end
avac:div(N)
avac=avac:double()
for i=1,n do
    padded=torch.zeros(3,30,30)
    padded[{1,{2,29},{2,29}}] = avac[{i}]
    padded[{2,{2,29},{2,29}}] = avac[{i}]
    padded[{3,{2,29},{2,29}}] = avac[{i}]
    ims[#ims+1] = zoom(padded,2)
end
rows[#rows+1] = image.toDisplayTensor(ims,3,n)
ims={}

print('some images')
Xs = X[{{177,179}}]
Zs = enc:forward(Xs)
Xsd = Xs:double():add(-Xs:min())
Xsd = Xsd:div(Xsd:max()) -- rescale to -1 1 range
Zsd = Zs:double()
for i = 1,3 do
    ims={}
    for j=1,n do
        padded=torch.zeros(3,30,30)
        padded[{1,{2,29},{2,29}}] = Zsd[{i,j}]
        padded[{2}] = Xsd[{i,1,{2,31},{2,31}}]*0.3 -- crop central part of orig Y as bg
        padded[{3}] = Xsd[{i,1,{2,31},{2,31}}]*0.3 -- crop central part of orig Y as bg
        ims[#ims+1] = zoom(padded,2)
    end
    rows[#rows+1] = image.toDisplayTensor(ims,3,n)
    ims={}
    for j=1,n do
        Zss = torch.CudaTensor(1,32,28,28):zero()
        Zss[{{},j}] = Zs[{{i},j}]
        Xr = dec:forward(Zss)[1]
        Xrd = Xr:double():add(-Xr:min())
        --Xrd = Xrd:div(Xrd:max())
        padded=torch.zeros(3,30,30)
        padded[{{},{4,27},{4,27}}] = Xrd
        --padded[{2}] = Xs[{i,1,{2,31},{2,31}}]*0.1 -- crop central part of orig Y as bg
        --padded[{3}] = Xs[{i,1,{2,31},{2,31}}]*0.1 -- crop central part of orig Y as bg
        ims[#ims+1] = zoom(padded,2)
    end
    rows[#rows+1] = image.toDisplayTensor(ims,3,n)

end

im = image.toDisplayTensor(rows,0,1)
image.save(string.format('filters_lam_%.6f.png', lambda_l1),im)

end
