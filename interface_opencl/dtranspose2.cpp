/*
 *   -- clMAGMA (version 1.0.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      April 2012
 *
 * @generated d Wed Oct 24 00:32:57 2012
 */

#include <stdio.h>

#include "magmablas.h"
#include "CL_MAGMA_RT.h"

#define DSIZE_1SHARED 32

magma_err_t
magma_dtranspose2(
	cl_mem odata, size_t offseto, int ldo, 
	cl_mem idata, size_t offseti, int ldi,
	int m, int n,
	magma_queue_t queue )
{
	if (m*n==0)
		return MAGMA_ERR_UNKNOWN;
	
	cl_int ciErrNum;                // Error code var
	cl_kernel ckKernel=NULL;
	
	ckKernel = rt->KernelPool["dtranspose3_32"];
	if (!ckKernel)
	{
		printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
		return MAGMA_ERR_UNKNOWN;
	}
	
	int nn = 0;
	ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(cl_mem), (void*)&odata   );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&offseto );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&ldo     );
	                            
	ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(cl_mem), (void*)&idata   );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&offseti );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&ldi     );
                                
	int m32 = (32-m%32)%32;     
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&m   );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&m32 );
	                            
	int n32 = (32-n%32)%32;     
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&n   );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&n32 );
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
		return MAGMA_ERR_UNKNOWN;
	}
	
	size_t GlobalWorkSize[2]={0,0}, LocalWorkSize[2]={0,0};
	
	LocalWorkSize[0] = DSIZE_1SHARED;
	LocalWorkSize[1] = 8;
	
	GlobalWorkSize[0] = ((m+31)/32)*LocalWorkSize[0];
	GlobalWorkSize[1] = ((n+31)/32)*LocalWorkSize[1];
	
	// launch kernel
	ciErrNum = clEnqueueNDRangeKernel(
		queue, ckKernel, 2, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL );
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
			__LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
		return MAGMA_ERR_UNKNOWN;
	}
	
	return MAGMA_SUCCESS;
}
