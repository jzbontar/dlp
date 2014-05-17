require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
--require 'torch_datasets'
require 'prof-torch'
require 'xlua'

Nepoch = Nepoch or 200
bs=128
hiddens = {32, 32, 64}
nout = 10
LR = 0.01
LRdecay = 0.1
NLRdec = 0
Wdecay = 0.0005


prof.tic('load')
--X,Y = torch_datasets:cifar(3)
X = torch.FloatTensor(torch.FloatStorage('/home/tom/datasets/cifar_whitened/X')):resize(60000, 3, 32, 32)
Y = torch.FloatTensor(torch.FloatStorage('/home/tom/datasets/cifar_whitened/y'))
X = X:cuda()
Y = Y:float():cuda()
N = 50000
Nte = 10000 -- last 10k of 60k
X_tr = X[{{1,N}}]
y_tr = Y[{{1,N}}]
X_te = X[{{N+1, N+Nte}}]
y_te = Y[{{N+1, N+Nte}}]
prof.toc('load')

prof.tic('model')
model = nn.Sequential()
model:set_bprop_min(2)
model:add(nn.SpatialZeroPadding(2,2,2,2))
model:add(nn.SpatialConvolution(3,hiddens[1],5,5))
model:add(nn.SpatialMaxPooling(2,2))
model:add(nn.Threshold(0,0))
-- todo contrastivenormalization
model:add(nn.SpatialZeroPadding(2,2,2,2))
model:add(nn.SpatialConvolution(hiddens[1],hiddens[2],5,5))
model:add(nn.SpatialMaxPooling(2,2))
model:add(nn.Threshold(0,0))
-- todo contrastivenormalization
model:add(nn.SpatialZeroPadding(2,2,2,2))
model:add(nn.SpatialConvolution(hiddens[2],hiddens[3],5,5))
model:add(nn.SpatialMaxPooling(2,2))
model:add(nn.Threshold(0,0))
model:add(nn.Reshape(4*4*hiddens[3]))
model:add(nn.Linear(4*4*hiddens[3], nout))
model:add(nn.LogSoftMax())
model:cuda()
print(model)
crit = nn.ClassNLLCriterion()
crit:cuda()
print(crit)
prof.toc('model')

function evaluate()
    -- train err
    err_tr = 0
    for i = 1,X_tr:size()[1],bs do
        X_b = X_tr[{{i,math.min(i+bs-1,X_tr:size()[1])}}]
        Y_b = y_tr[{{i,math.min(i+bs-1,X_tr:size()[1])}}]
        Ypred = model:forward(X_b):double()
        _, ix = torch.max(Ypred, 2)
        err_tr = err_tr+torch.ne(ix,Y_b:double():long()):sum()
    end
    err_tr = err_tr/X_tr:size()[1]
    Etr[#Etr+1] = err_tr
    -- test set 
    err_te = 0
    for i = 1,X_te:size()[1],bs do
        X_b = X_te[{{i,math.min(i+bs-1,X_te:size()[1])}}]
        Y_b = y_te[{{i,math.min(i+bs-1,X_te:size()[1])}}]
        Ypred = model:forward(X_b):double()
        _, ix = torch.max(Ypred, 2)
        err_te = err_te+torch.ne(ix,Y_b:double():long()):sum()
    end
    err_te = err_te/X_te:size()[1]
    Ete[#Ete+1] = err_te
end

prof.tic('train')
print('start training phase')
Etr= {}
Ete= {}
evaluate()
epoch = 0
print('Epoch ' .. epoch .. ' train_err=' .. Etr[#Etr] .. ' test_err=' .. Ete[#Ete])
par, gpar = model:getParameters()
momentum = gpar:clone():zero()
for epoch = 1,Nepoch do
    prof.tic('epoch')
    alpha = LR --/ (1 + LRdecay*epoch)
    for i = 1,X_tr:size()[1],bs do
        prof.tic('batch')
        --xlua.progress(i, X_tr:size()[1])
        X_b = X_tr[{{i,math.min(i+bs-1,X_tr:size()[1])}}]
        Y_b = y_tr[{{i,math.min(i+bs-1,X_tr:size()[1])}}]
        model:zeroGradParameters()
        go = crit:backward(model:forward(X_b), Y_b)
        model:backward(X_b, go)
        -- Simple training no weigh decay
        --model:updateParameters(alpha)
        -- Momentum and weight decay
        momentum:mul(0.9)
        momentum:add(-Wdecay*alpha, par)
        momentum:add(-alpha, gpar)
        par:add(momentum)
        prof.toc('batch')
    end
    prof.toc('epoch')
    prof.tic('evaluate')
    evaluate()
    print('Epoch ' .. epoch .. ' train_err=' .. err_tr .. ' test_err=' .. err_te)
    prof.toc('evaluate')
    if Ete[#Ete] >= Ete[#Ete-1] and Ete[#Ete] >= Ete[#Ete-2] then 
        if NLRdec ==3 then  break end
        LR = LR*LRdecay
        NLRdec = NLRdec+1
        print("Adjusted Learning Rate for " .. NLRdec .. " time, LR = "..LR)
    end
end
prof.toc('train')

prof.dump()
torch.save('1_nosae.net', model)
