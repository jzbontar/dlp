extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <stdio.h>
#include <assert.h>


__global__ void shrink_kernel(float *x, float amount, int size_of_x)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size_of_x) {
		if (x[id] > amount) {
			x[id] -= amount;
		} else if (x[id] < -amount) {
			x[id] += amount;
		} else {
			x[id] = 0;
		}	
	}
}


int shrink(lua_State *L)
{
	THCudaTensor *x = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	float amount = luaL_checknumber(L, 2);

	int size_of_x = THCudaTensor_nElement(x);

	int tb = 128;
	shrink_kernel<<< (size_of_x - 1) / tb + 1, tb >>>(THCudaTensor_data(x), amount, size_of_x);

	return 0;
}


static const struct luaL_Reg funcs[] = {
	{"shrink_from_lua", shrink},
	{NULL, NULL}
};

extern "C" int luaopen_libdlp(lua_State *L) {
	luaL_openlib(L, "dlp", funcs, 0);
	return 1;
}
