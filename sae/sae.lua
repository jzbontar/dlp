require 'image'
require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
--require 'jzt'
require 'L1CostMan'
require 'prof-torch'

cutorch.setDevice(1)
torch.manualSeed(42)

Nepoch = Nepoch or 3
batch_size = 128
lambda_l1= 0.001 -- lambda. Beta=1
LR = 0.00001 -- learning rate
Wdecay = 0.0005

prof.tic('load')
--X,Y = torch_datasets:cifar(3)
X = torch.FloatTensor(torch.FloatStorage('/home/tom/datasets/cifar_whitened/X')):resize(60000, 3, 32, 32):cuda()
--N = N or 50000
N = N or 1024
X = X[{{1,N}}]
prof.toc('load')

enc = nn.Sequential{bprop_min=1}
enc:add(nn.SpatialConvolutionRing2(3, 64, 5, 5))
enc:add(nn.Threshold())
enc:cuda()

enc_sparse = nn.Sequential()
--enc_sparse:add(nn.SpatialLPPooling(64, 2, 2, 2, 2, 2)) -- gives nans
enc_sparse:add(nn.SpatialMaxPooling(2,2,2,2))
enc_sparse:cuda()

dec = nn.Sequential()
dec:add(nn.SpatialZeroPadding(4, 4, 4, 4))
dec:add(nn.SpatialConvolutionRing2(64, 16, 5, 5))
-- note: 16 because Ring2 requires it. only first 3 are used.
dec:cuda()

reconstruct_cost = nn.MSECriterion():cuda()
l1cost = nn.L1CostMan():cuda()

--layers = {}
--for _, net in ipairs({enc,dec,enc_sparse}) do
   --for i = 1,net:size() do
      --if net:get(i).weight then
         --layer = net:get(i)
         --layer.weight_v = torch.CudaTensor():resizeAs(layer.weight):zero()
         --layer.bias_v = torch.CudaTensor():resizeAs(layer.bias):zero()
         --layer.gradWeight:zero()
         --layer.gradBias:zero()
         --layers[#layers + 1] = layer
      --end
   --end
--end
-- print(('Number of layers with parameters: %d'):format(#layers))

function plotweights(filename, wt)
    local table = {}
    for i = 1,wt:size()[1] do
        for j = 1,wt:size()[2] do 
            table[#table+1] = wt[{i,j}]
        end
    end
    image.save(filename, image.toDisplayTensor(table, 2, torch.floor(torch.sqrt(#table))))
end

print('train')
prof.tic('train')
epar, egpar = enc:getParameters()
emom = egpar:clone():zero()
dpar, dgpar = dec:getParameters()
dmom = dgpar:clone():zero()
for epoch = 1,Nepoch do
   -- train
   tic= prof.time()
   prof.tic('epoch')
   l1_tot = 0
   rec_tot = 0
   for t = 1,N- batch_size, batch_size do
      prof.tic('batch')
      X_batch = X:narrow(1, t, batch_size)
      enc:zeroGradParameters()
      --enc_sparse:zeroGradParameters() -- has no params
      dec:zeroGradParameters()

      enc:forward(X_batch)

      enc_sparse:forward(enc.output)
      l1_tot = l1_tot + l1cost:forward(enc_sparse.output)
      l1cost:backward(enc_sparse.output)
      enc_sparse:backward(enc.output, l1cost.gradInput)
      --print ('l1 gradInput: ' .. l1cost.gradInput:abs():mean() .. ' / maxpool gradInput: ' .. enc_sparse.gradInput:abs():mean())

      dec:forward(enc.output)
      recout = dec.output:narrow(2, 1, 3) -- select first 3 maps from 16   
      rec_tot = rec_tot + reconstruct_cost:forward(recout, X_batch)
      reconstruct_cost:backward(recout, X_batch)
      dec:backward(enc.output, reconstruct_cost.gradInput)

      loss = reconstruct_cost.output + lambda_l1 * l1cost.output

      --print('sp '.. enc_sparse.gradInput:abs():mean() .. ', rec ' .. dec.gradInput:abs():mean())
      -- combine l1 and reconstruction backprop
      --gradInput = enc_sparse.gradInput:add(dec.gradInput)
      gradInput = dec.gradInput:add(lambda_l1, enc_sparse.gradInput)
      enc:backward(X_batch, gradInput)
      
      -- Momentum and weight decay                                                                              
      emom:mul(0.9)                                                                                         
      --emom:add(-Wdecay*alpha, epar) --weight decay on encoder doesnt make sense
      emom:add(-LR, egpar) 
      epar:add(emom)
      dmom:mul(0.9)
      dmom:add(-LR*Wdecay, dpar)
      dmom:add(-LR,dgpar)
      dpar:add(dmom)

      prof.toc('batch')
   end
   print(string.format('gradInputs: sp %.4f / rec %.4f' , enc_sparse.gradInput:abs():mean(), dec.gradInput:abs():mean()))
   print(string.format('average enc weight: %.4f / average dec weight: %.4f',epar:abs():mean(), dpar:abs():mean()))
   print(string.format('epoch %d, time %.2f, L1 %.2f, reconstr %.2f', epoch, prof.time()-tic, l1_tot, rec_tot))
   wt = enc:get(1):parameters()[1]
   --image.save(string.format('filters_%03d.jpg', epoch), image.toDisplayTensor({wt[{{},1}], wt[{{},2}], wt[{{},3}]}, 2, 14))
   plotweights(string.format('filters_%03d.jpg', epoch), wt)
   prof.toc('epoch')
end
prof.toc('train')
prof.dump()
