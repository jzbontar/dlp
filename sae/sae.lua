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
torch.manualSeed(42)

nhidden = 32
batch_size = 16
lambda_l1= 0 -- 1e-5 --0.0001 -- lambda. Beta=1

--LR = 5e-6 -- learning rate
lrd = 5e-4
lre = 5e-4

prof.tic('load')
--X,Y = torch_datasets:cifar(3)
X = torch.FloatTensor(torch.FloatStorage('/home/tom/datasets/cifar_whitened/X')):resize(60000, 3, 32, 32):cuda()
collectgarbage() -- in case of re-run several times

--N = N or 50000
N = N or 10240
Nepoch = Nepoch or 9
X = X[{{1,N}}]
prof.toc('load')

--enc = nn.Sequential{bprop_min=1}
enc = nn.Sequential()
--enc:add(nn.SpatialZeroPadding(2,2,2,2))
--enc:add(nn.SpatialConvolutionRing2(3, 32, 5, 5))
enc:add(nn.SpatialConvolution(3, 32, 5, 5))
enc:add(nn.Threshold())
enc:cuda()

enc_sparse = nn.Sequential()
--enc_sparse:add(nn.SpatialLPPooling(32, 2, 2, 2, 2, 2)) -- gives nans
enc_sparse:add(nn.SpatialMaxPooling(2,2,2,2))
enc_sparse:cuda()

dec = nn.Sequential()
--dec:add(nn.SpatialZeroPadding(2, 2, 2, 2))
dec:add(nn.SpatialConvolution(32, 3, 5, 5))
--dec:add(nn.SpatialConvolutionRing2(32, 16, 5, 5))
-- note: 16 because Ring2 requires it. only first 3 are used.
-- initialize decoder with flipped version of encoder weights 
for i = 1,3 do
    for j=1,32 do
        for k=1,5 do
            for l = 1,5 do
                dec:get(1).weight[{i,j,6-k,6-l}] = enc:get(1).weight[{j,i,k,l}]
                --dec:get(1).weight[{i,j,k,l}] = enc:get(1).weight[{j,i,k,l}]
            end
        end
    end
end
dec:cuda()

reconstruct_cost = nn.MSECriterion():cuda()
l1cost = nn.L1CostMan():cuda()

picpadder = nn.SpatialZeroPadding(4,4,4,4):cuda()

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
    local sca = torch.Tensor(25,25)
    for i = 1,wt:size()[1] do
        for j = 1,wt:size()[2] do 
            for k = 1,wt:size(3) do
                for l = 1,wt:size()[4] do
                    sca[{{k*5-4,k*5},{l*5-4,l*5}}] = wt[{i,j,k,l}]
                end
            end
            table[#table+1] = sca:clone()
        end
    end
    image.save(filename, image.toDisplayTensor(table, 2, torch.floor(torch.sqrt(#table))))
end

print('train')
prof.tic('train')
epar, egpar = enc:getParameters()
dpar, dgpar = dec:getParameters()
lam_heur = {}
lrd_heur = {}
lre_heur = {}
introsp = false


wt = enc:get(1).weight
--image.save(string.format('filters_%03d.jpg', epoch), image.toDisplayTensor({wt[{{},1}], wt[{{},2}], wt[{{},3}]}, 2, 14))
plotweights(string.format('filters_%03d.png', 0), wt)
for epoch = 1,Nepoch do
   -- train
   tic= prof.time()
   prof.tic('epoch')
   l1_coll = {}
   rec_coll = {}
   units_on = {}
   for t = 1,N- batch_size, batch_size do
      prof.tic('batch')
      X_batch = X:narrow(1, t, batch_size)
      --enc:zeroGradParameters()
      --enc_sparse:zeroGradParameters() -- has no params
      --dec:zeroGradParameters()
      egpar:zero()
      dgpar:zero()

      enc:forward(X_batch)
      if introsp then
          units_on[#units_on + 1] = torch.sign(enc.output:double()):sum() / enc.output:nElement()
      end

      enc_sparse:forward(enc.output)
      l1cost:forward(enc_sparse.output)
      l1_coll[#l1_coll + 1] = l1cost.output/enc_sparse.output:nElement()
      l1cost:backward(enc_sparse.output)
      enc_sparse:backward(enc.output, l1cost.gradInput)
      -- DONTUSE print ('l1 gradInput: ' .. l1cost.gradInput:abs():mean() .. ' / maxpool gradInput: ' .. enc_sparse.gradInput:abs():mean())

      dec:forward(enc.output)
      --recout = dec.output:narrow(2, 1, 3) -- select first 3 nfmaps from 16   
      recout = dec.output
      center = X_batch:narrow(3,5,24):narrow(4,5,24)
      reconstruct_cost:forward(recout, center)
      rec_coll[#rec_coll + 1] = reconstruct_cost.output
      reconstruct_cost:backward(recout, center)
      dec:backward(enc.output, reconstruct_cost.gradInput)

      -- based on current input gradients and param size / gradients, get heuristic LR and lambda
      if introsp then 
          lam_heur[#lam_heur+1] = torch.abs(dec.gradInput:double()):mean() / torch.abs(enc_sparse.gradInput:double()):mean()
          lrd_heur[#lrd_heur+1] = 0.01 * torch.abs(dpar:double()):mean() / torch.abs(dgpar:double()):mean()
          --print('Reasonable lambda: ', lam_heur[#lam_heur])
          --print('Reasonable LR dec: ', lrd_heur[#lrd_heur])
      end

      -- DONTUSE print('sp '.. enc_sparse.gradInput:abs():mean() .. ', rec ' .. dec.gradInput:abs():mean())
      -- combine l1 and reconstruction backprop
      --gradInput = enc_sparse.gradInput:add(dec.gradInput)
      --gradInput = dec.gradInput:add(lambda_l1, enc_sparse.gradInput)
      gradInput = dec.gradInput
      enc:backward(X_batch, gradInput)
      if introsp then
          lre_heur[#lre_heur+1] = 0.01 * torch.abs(epar:double()):mean() / torch.abs(egpar:double()):mean()
          --print('Reasonable LR enc: ', lre_heur[#lre_heur])
      end

      -- update parameters
      epar:add(-lre, egpar)
      dpar:add(-lrd, dgpar)
      --epar:add(1,torch.CudaTensor(2432):normal())
      --dpar:add(1,torch.CudaTensor(12816):normal())
      --print(reconstruct_cost.output, torch.abs(egpar:double()):mean(), torch.abs(dgpar:double()):mean())

      -- normalize decoder kernels
      wt = dec:get(1).weight
      for nfmap = 1,nhidden do
          wt[{{1,3},nfmap}]:div(wt[{{1,3},nfmap}]:norm(2))
      end

      prof.toc('batch')
   end
   prof.toc('epoch')
   print(string.format('epoch %d, time %.2f, L1 %.2f, reconstr %.2f', epoch, prof.time()-tic, torch.Tensor(l1_coll):mean(), torch.Tensor(rec_coll):mean()))
   print(string.format('gradInputs: sp %.4f / rec %.4f', torch.abs(enc_sparse.gradInput:double()):mean(), torch.abs(dec.gradInput:double()):mean()))
   print(string.format('average enc weight: %.4f / average dec weight: %.4f',torch.abs(epar:double()):mean(), torch.abs(dpar:double()):mean()))
   if introsp then
       print(string.format('Heuristics: lambda %.6f / enc LR %.6f / dec LR %.6f, units on %.4f',torch.Tensor(lam_heur)[{{-30,-1}}]:mean(),torch.Tensor(lre_heur)[{{-30,-1}}]:mean(),torch.Tensor(lrd_heur)[{{-40,-1}}]:mean(), torch.Tensor(units_on):mean()))
   end
   --print(dec:get(1).weight[{1,1}])
   print("=====")
   --image.save(string.format('filters_%03d.jpg', epoch), image.toDisplayTensor({wt[{{},1}], wt[{{},2}], wt[{{},3}]}, 2, 14))
   wt = enc:get(1).weight
   plotweights(string.format('filters_%03d.png', epoch), wt)
   if epoch==1 then
       imgs = {}
       imgs[1] = image.toDisplayTensor(X_batch,2,1) 
   end
   imgs[#imgs+1] = image.toDisplayTensor(picpadder:forward(recout),2,1)
   --image.save(string.format('orig_%02d.png',epoch), image.toDisplayTensor(X_batch,2))
   --image.save(string.format('reco_%02d.png',epoch), image.toDisplayTensor(recout,2))
   image.save('reconstruction.png', image.toDisplayTensor(imgs, 0, #imgs))
end
prof.toc('train')
prof.dump()
