require 'libdlp'
require 'cutorch'

x = torch.CudaTensor(10):normal()

print(x)
dlp.shrink_from_lua(x, 1)
print(x)
