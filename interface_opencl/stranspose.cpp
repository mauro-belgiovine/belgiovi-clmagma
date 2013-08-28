/*
 *   -- clMAGMA (version 1.0.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      April 2012
 *
 * @generated s Wed Oct 24 00:32:56 2012
 */

#include <stdio.h>

#include "magmablas.h"
#include "CL_MAGMA_RT.h"

#define SSIZE_1SHARED 32

magma_err_t
magma_stranspose(
	cl_mem odata, int offseto, int ldo, 
	cl_mem idata, int offseti, int ldi,
	int m, int n, 
	magma_queue_t queue )
{
	if (m*n==0)
		return MAGMA_SUCCESS;
	
	cl_int ciErrNum;                // Error code var
	cl_kernel ckKernel=NULL;
	
	ckKernel = rt->KernelPool["stranspose_32"];
	if (!ckKernel)
	{
		printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
		return MAGMA_ERR_UNKNOWN;
	}
	
	int nn = 0;
	ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(cl_mem), (void*)&odata   );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&offseto );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&ldo     );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_mem), (void*)&idata   );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&offseti );
	ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&ldi     );
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: clSetKernelArg at %d in file %s, %s!\n", __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
		return MAGMA_ERR_UNKNOWN;
	}
	
	size_t GlobalWorkSize[2]={0,0}, LocalWorkSize[2]={0,0};
	//BELGIOVINE
	magma_device_t device_id;
	size_t local = 0;
	clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(magma_device_t), &device_id, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
	    printf("Error %d: Failed to retrieve kernel work group info!\n", ciErrNum);
	    return MAGMA_ERR_UNKNOWN;
	}
	
	ciErrNum = clGetKernelWorkGroupInfo(ckKernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
	    printf("Error %d: Failed to retrieve kernel work group info!\n", ciErrNum);
	    return MAGMA_ERR_UNKNOWN;
	}
	
	//printf("[stranspose_32] ");
	
	LocalWorkSize[0] = SSIZE_1SHARED;
	//TEST	MODIFICHE IN control/get_nb_fermi
	//get_sgetrf_nb()!!!
	LocalWorkSize[1] = 8;
	///LocalWorkSize[1] = 16;
	
	GlobalWorkSize[0] = (m/32)*LocalWorkSize[0];
	GlobalWorkSize[1] = (n/32)*LocalWorkSize[1];
	///GlobalWorkSize[1] = (n/64)*LocalWorkSize[1];
	
	//printf("m = %d, n = %d, GlobalWorkSize[0] = %d, GlobalWorkSize[1] = %d\n", m, n, GlobalWorkSize[0], GlobalWorkSize[1]);
	//BELGIOVINE end
	
	// launch kernel
	ciErrNum = clEnqueueNDRangeKernel(
		queue, ckKernel, 2, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\" (%d)\n",
			__LINE__, __FILE__, rt->GetErrorCode(ciErrNum), ciErrNum);
		return MAGMA_ERR_UNKNOWN;
	}
	
	return MAGMA_SUCCESS;
}
