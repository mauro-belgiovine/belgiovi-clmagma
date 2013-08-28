/*
 *   -- clMAGMA (version 1.0.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      April 2012
 *
 * @author Mark Gates
 */

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "CL_MAGMA_RT.h"
//#define HAVE_clAmdBlas
//#define DEBUG_V

#ifdef HAVE_clAmdBlas

// ========================================
// globals
cl_platform_id gPlatform;
cl_context     gContext;

// Run time global variable used for LU
CL_MAGMA_RT *rt;

// ========================================
// initialization

//BELGIOVINE
magma_err_t
magma_init()
{
    cl_int err;
    
    /*err = clGetPlatformIDs( 1, &gPlatform, NULL );
    assert( err == 0 );        

    cl_device_id devices[ MagmaMaxGPUs ];
    cl_uint num;
    err = clGetDeviceIDs( gPlatform, CL_DEVICE_TYPE_GPU, MagmaMaxGPUs, devices, &num );
    printf("err = %d\n", err);
    assert( err == 0 );
   
    cl_context_properties properties[3] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties) gPlatform, 0 };
    gContext = clCreateContext( properties, num, devices, NULL, NULL, &err );
    assert( err == 0 );*/
    // Init clAmdBlas
    err = clAmdBlasSetup();
    assert( err == 0 );

    // Initialize kernels related to LU
    rt = CL_MAGMA_RT::Instance();
    //rt->Init(gPlatform, gContext);
    rt->InitAll();
    
    gPlatform = rt->GetPlatform();
    
    gContext = rt->GetContext();
    
 
    return err;
}


// --------------------
magma_err_t
magma_finalize()
{
    cl_int err;
    clAmdBlasTeardown();
    //err = clReleaseContext( gContext );

    // quit the RT
    rt->Quit();

    return err;
}


// ========================================
// memory allocation
// #include "CL/cl_ext.h"
// ...
// *ptrPtr = clCreateBuffer( gContext, CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD, size, NULL, &err );
magma_err_t
magma_malloc( magma_ptr* ptrPtr, size_t size )
{
    cl_int err;
    *ptrPtr = clCreateBuffer( gContext, CL_MEM_READ_WRITE, size, NULL, &err );
    return err;
}

//crea un sub-buffer del buffer dato
magma_err_t
magma_sub_buffer(magma_ptr* subBuff, magma_ptr dA_src, size_t dA_offset){
  
  cl_buffer_region buffer_create_info;
  cl_int err;
  size_t dA_size;
  
  err = clGetMemObjectInfo(dA_src, CL_MEM_SIZE, sizeof(dA_size), (void *) &dA_size, NULL);
  
  if(err == CL_SUCCESS){
    buffer_create_info.origin = dA_offset;
    buffer_create_info.size = dA_size;
  
    *subBuff = clCreateSubBuffer(
	dA_src, CL_MEM_READ_WRITE,
 	CL_BUFFER_CREATE_TYPE_REGION,
 	(const void *) &buffer_create_info,
 	&err);
  }
  
  return err;
}
//BELGIOVINE end
// --------------------
magma_err_t
magma_free( magma_ptr ptr )
{
    cl_int err = clReleaseMemObject( ptr );
    return err;
}

// --------------------
magma_err_t
magma_malloc_host( void** ptrPtr, size_t size )
{
    *ptrPtr = malloc( size );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    else {
        return MAGMA_SUCCESS;
    }
}

// --------------------
magma_err_t
magma_free_host( void* ptr )
{
    free( ptr );
    return MAGMA_SUCCESS;
}


// ========================================
// device & queue support
magma_err_t
magma_get_devices(
	magma_device_t* devices,
	magma_int_t     size,
	magma_int_t*    numPtr )
{
    cl_int err;
    //err = clGetDeviceIDs( gPlatform, CL_DEVICE_TYPE_GPU, 1, size, devices, num );
    size_t n;
    //devices = (magma_device_t *) malloc(sizeof(magma_device_t)*size);
    
    err = clGetContextInfo(
        gContext, CL_CONTEXT_DEVICES,
        size*sizeof(magma_device_t), devices, &n );
    *numPtr = n / sizeof(magma_device_t);
    
    //free(devices);
    //devices = rt->GetDevicePtr();
    
    return err;
}

// --------------------
magma_err_t
magma_queue_create( magma_device_t device, magma_queue_t* queuePtr )
{
    assert( queuePtr != NULL );
    cl_int err;
    *queuePtr = clCreateCommandQueue( gContext, device, 0, &err );
    return err;
}

//BELGIOVINE
//Get queue from global cl_platform class
magma_err_t
magma_get_queue(magma_device_t device, magma_queue_t* queuePtr){
  
    assert( queuePtr != NULL );
    cl_int err;
    
    magma_device_t * devices = rt->GetDevicePtr();
    
    for(uint i = 0; i < rt->GetNumDevices(); i++){
      
      if(devices[i] == device){
	*queuePtr = rt->GetCommandQueue(i);
	return MAGMA_SUCCESS;
      }
      
    }
    
    return MAGMA_ERR_UNKNOWN;
  
}

/* ------------- NOTE ------------------------------------
 * Questa funzione permette di cambiare piattaforma e tipo di device
 * da utilizzare per le computazioni INDIPENDENTI. Al momento è possibile cambiare
 * il contesto esecutivo solo per le chiamate di magma_nomefunzione()
 * e non all'interno delle stesse, in quanto dovrebbero essere riallocati i dati per
 * il device di riferimento
 */
magma_err_t magma_switch_platform(cl_device_type device_type, uint numDevices, magma_queue_t* queuePtr)
{
    assert(numDevices > 0);
    int num = 0;
    cl_int err;
    magma_device_t devicePtr[numDevices];
    
    cl_platform_id ret = NULL;
    
    for(uint i = 0; i < rt->GetNumPlatform(); i++){
      ret = rt->SetPlatform(i, device_type);
      if (ret != NULL) break; 
    }
    
    if(ret == NULL){
      fprintf( stderr, "SetPlatform failed: %d\n", err );
      return MAGMA_ERR_UNKNOWN;
    }
    
    gPlatform = rt->GetPlatform();
    gContext = rt->GetContext();
    
    err = magma_get_devices( &devicePtr[0], numDevices, &num );
    
    if ( err != MAGMA_SUCCESS || num < 1 ) {
      fprintf( stderr, "magma_get_devices failed: %d\n", err );
      return err;
    }
    
    for(int i = 0; i < numDevices; i++){
      err = magma_get_queue( devicePtr[i], &queuePtr[i] );
      if ( err != MAGMA_SUCCESS ) {
	fprintf( stderr, "magma_queue_create failed on device %d: %d\n", i, err );
	return err;
      }
    }
    
    return err;
    
}
//BELGIOVINE end

// --------------------
magma_err_t
magma_queue_destroy( magma_queue_t  queue )
{
    cl_int err = clReleaseCommandQueue( queue );
    return err;
}

// --------------------
magma_err_t
magma_queue_sync( magma_queue_t queue )
{
    cl_int err = clFinish( queue );
    return err;
}


// ========================================
// event support
magma_err_t
magma_event_create( magma_event_t* event )
{
    printf( "%s not implemented\n", __func__ );
    return 0;
}

magma_err_t
magma_event_destroy( magma_event_t event )
{
    magma_err_t err;
    err = clReleaseEvent(event);
    return err;
}

magma_err_t
magma_event_record( magma_event_t event, magma_queue_t queue )
{
    printf( "%s not implemented\n", __func__ );
    return 0;
}

magma_err_t
magma_event_query( magma_event_t event )
{
    printf( "%s not implemented\n", __func__ );
    return 0;
}

magma_err_t
magma_event_sync(magma_event_t event )
{
    cl_int err = clWaitForEvents(1, &event);
    return err;
}

#endif // HAVE_clAmdBlas

// -------- STARPU FUNCTIONS ---------
/*
magma_err_t
magma_setmatrix_PU(float const *hA_src, magmaFloat_ptr *dA_dst, size_t dA_offset)
{
  *dA_dst = STARPU_MATRIX_GET_DEV_HANDLE((void *) hA_src);
  return MAGMA_SUCCESS;
}
*/
