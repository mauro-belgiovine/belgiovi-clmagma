#include "CL_MAGMA_RT.h"

#include <fstream>
#include <iostream>
#include <string.h>
#include <sys/stat.h>
#include <stdexcept>      // std::out_of_range
#include <errno.h>

#include <cl.h>

using std::string;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::vector;


// define number of command queues to create
#define QUEUE_COUNT 1


//cl_platform class: function definition

//constructor for cl_platform class, with void argument
inline cl_platform::cl_platform(){
	n_gpu = n_cpu = n_acc = 0;
	platform = NULL;
	gpu_devices = NULL;
	gpu_context = NULL;
	gpu_queue = NULL;
	cpu_devices = NULL;
	cpu_context = NULL;
	cpu_queue = NULL;
	acc_devices = NULL;
	acc_context = NULL;
	acc_queue = NULL;
}

//deep copy constructor for cl_platform class
inline cl_platform::cl_platform(const cl_platform &old_platform){
	
	n_gpu = old_platform.n_gpu;
	n_cpu = old_platform.n_cpu;
	n_acc = old_platform.n_acc;
	
	platform = old_platform.platform;
	
	gpu_context = old_platform.gpu_context;
	gpu_devices = new cl_device_id[n_gpu];
	for(uint y = 0; y < n_gpu; y++) gpu_devices[y] = old_platform.gpu_devices[y];
	gpu_queue = new cl_command_queue[n_gpu];
	for(uint y = 0; y < n_gpu; y++) gpu_queue[y] = old_platform.gpu_queue[y];
	
	cpu_context = old_platform.cpu_context;
	cpu_devices = new cl_device_id[n_cpu];
	for(uint y = 0; y < n_cpu; y++) cpu_devices[y] = old_platform.cpu_devices[y];
	cpu_queue = new cl_command_queue[n_cpu];
	for(uint y = 0; y < n_cpu; y++) cpu_queue[y] = old_platform.cpu_queue[y];
	
	acc_context = old_platform.acc_context;
	acc_devices = new cl_device_id[n_acc];
	for(uint y = 0; y < n_acc; y++) acc_devices[y] = old_platform.acc_devices[y];
	acc_queue = new cl_command_queue[n_acc];
	for(uint y = 0; y < n_acc; y++) acc_queue[y] = old_platform.acc_queue[y];
}

inline cl_platform::~cl_platform(){
	
	for(uint y = 0; y < n_gpu; y++) clReleaseCommandQueue(gpu_queue[y]);
	if (gpu_queue)	delete [] gpu_queue;
	if(gpu_devices)	delete [] gpu_devices;
	if(gpu_context)	clReleaseContext(gpu_context);
	for(uint y = 0; y < n_cpu; y++) clReleaseCommandQueue(cpu_queue[y]);
	if (cpu_queue)	delete [] cpu_queue;
	if(cpu_devices)	delete [] cpu_devices;
	if(cpu_context)	clReleaseContext(cpu_context);
	for(uint y = 0; y < n_acc; y++) clReleaseCommandQueue(acc_queue[y]);
	if (acc_queue)	delete [] acc_queue;
	if(acc_devices)	delete [] acc_devices;
	if(acc_context)	clReleaseContext(acc_context);
}

//------------------------------------------------------------------------------------------------//

/* 
 * constructor
 */
CL_MAGMA_RT::CL_MAGMA_RT()
{
	HasBeenInitialized = false;
	ciDeviceCount = 0;
	ceEvent = NULL;
	ckKernel = NULL;
	
	cpPlatform = NULL;
	cdDevices = NULL;
	commandQueue = NULL;
	cxGPUContext = NULL;
	dev_type = 0;
	platform_id = 0;
	
}

/* 
 * destructor 
 */
CL_MAGMA_RT::~CL_MAGMA_RT()
{
	
	if (!HasBeenInitialized)
		return;

	// Cleanup allocated objects
	if (commandQueue)	delete [] commandQueue;
	if(cdDevices)		delete [] cdDevices;
	if(cxGPUContext)	clReleaseContext(cxGPUContext);
	
	
	if(ceEvent)	clReleaseEvent(ceEvent);  
	if(ckKernel)	clReleaseKernel(ckKernel);
		
	if(cpPlatforms.size() > 0) {
	  cpPlatforms.clear();
	}
	
}

cl_platform_id CL_MAGMA_RT::GetPlatform()
{
	return cpPlatform;
}

cl_platform_id CL_MAGMA_RT::SetPlatform(uint platformid, cl_device_type device_type)
{
  
  cl_platform platform;
  cl_platform_id out_p = NULL;
  
  if((cpPlatforms.size() > 0) && HasBeenInitialized){
    try {
      platform = cpPlatforms.at(platformid);    // vector::at throws an out-of-range
    }
    catch (const std::out_of_range& oor) {
      std::cerr << "Platform not found: " << oor.what() << ". Trying to set default (0) platform.\n";
      platformid = 0;
      platform = cpPlatforms.at(0);
    }
    
    switch(device_type){
      
      case CL_DEVICE_TYPE_CPU:
		if(platform.n_cpu > 0){
			cxGPUContext  = platform.cpu_context;
			commandQueue  = platform.cpu_queue;
			cdDevices = platform.cpu_devices;
			ciDeviceCount = platform.n_cpu;
			dev_type = CL_DEVICE_TYPE_CPU;
			platform_id = platformid;
			out_p = platform.platform;
		}else {
			std::cerr << "[WARNING] CPUs not found in platform " << platformid << ": you should try setting default GPUs Platform\n";
		}
		break;
	
      default:
		if(platform.n_gpu > 0){
			cxGPUContext  = platform.gpu_context;
			commandQueue  = platform.gpu_queue;
			cdDevices = platform.gpu_devices;
			ciDeviceCount = platform.n_gpu;
			dev_type = CL_DEVICE_TYPE_GPU;
			platform_id = platformid;
			out_p = platform.platform;
		}else{
			std::cerr << "[WARNING] GPUs not found in platform "<< platformid << ".\n";
		}
    }
    
    //NOTE se abbiamo generato la mappa dei kernel, sostituiamola con quella della piattaforma inizializzata
    if(!KernelPool.empty()) KernelPool = BigKernelPool[platformid][dev_type];
    
  } else {
     std::cerr << "[ERROR] Platforms not initialized yet. Aborting\n";
  }
  
  return out_p;
  
}

size_t CL_MAGMA_RT::GetNumPlatform()
{
  return cpPlatforms.size();
}

size_t CL_MAGMA_RT::GetNumDevices()
{
  return (size_t) ciDeviceCount;
}

cl_command_queue CL_MAGMA_RT::GetCommandQueue(int queueid)
{
	return (queueid>=ciDeviceCount) ? NULL : commandQueue[queueid];
}

magma_queue_t* CL_MAGMA_RT::GetQueuePtr()
{
	return commandQueue;
}

cl_device_id * CL_MAGMA_RT::GetDevicePtr()
{
	return cdDevices;
}

cl_context CL_MAGMA_RT::GetContext()
{
	return cxGPUContext;
}
//edit
bool CL_MAGMA_RT::initDevices(const cl_platform_id src_platform, cl_device_id** devices, cl_context* context,  cl_uint* num, cl_command_queue** queue, cl_device_type device_type, cl_uint max_ndev, cl_int *ciErrNum, char* label){
  
  char chBuffer[512];
  cl_uint n_device = 0;
  //check args
  if(max_ndev < 1){
    printf("ERROR! Param max_ndev must be at least 1 or higher. %d is given.\n", max_ndev);
    return false;
  }
  
  *ciErrNum = clGetDeviceIDs (src_platform, device_type, 0, NULL, &n_device);
  if (n_device == 0){
      printf(" No %s devices in context found supporting OpenCL (return code %i)\n", label, *ciErrNum);
      
  }else if (*ciErrNum != CL_SUCCESS){
    
    printf(" Error %i in clGetDeviceIDs (%s) call !!!\n\n", *ciErrNum, label);
    return false;
  }else{// Get and log the OpenCL device ID's
	 
    printf(" %u %s devices found supporting OpenCL:\n\n", n_device, label);
    *devices = new cl_device_id[n_device];
	
    if(new_devices == NULL){  
		printf(" Failed to allocate memory for devices !!!\n\n");
		return false;
    }else{
		
	*num = n_device;
	*ciErrNum = clGetDeviceIDs (src_platform, device_type, n_device, (cl_device_id *) *devices, NULL);
	
	if (*ciErrNum == CL_SUCCESS){
	  //Create a context for the devices
	  cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) src_platform, 0 };
	  *context = clCreateContext( properties, n_device, *devices, NULL, NULL, ciErrNum );

	  if (*ciErrNum != CL_SUCCESS){
	    printf("Error %i in clCreateContext (%s) call !!!\n\n", *ciErrNum, label);
		
	    return false;
	  } else {
	
	    // show info for each device in the context and init queue
		//CHECK ------> http://dhruba.name/2012/10/14/opencl-cookbook-how-to-leverage-multiple-devices-in-opencl/
		*queue = (magma_queue_t *) new magma_queue_t[n_device];
		
		if(*queue == NULL){
			printf(" Failed to allocate memory for devices !!!\n\n");
			return false;
		}
		
	    for(unsigned int y = 0; y < n_device; y++ ) {
			cl_uint queue_count;
			clGetDeviceInfo(new_devices[y], CL_DEVICE_NAME, sizeof(chBuffer), &chBuffer, NULL);
			printf("\t- %s Device %s\n", label, chBuffer);
			// create command queue
			*queue[y] = clCreateCommandQueue(*context, *devices[y], CL_QUEUE_PROFILING_ENABLE, ciErrNum);
			clGetCommandQueueInfo(*queue[y], CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &queue_count, NULL);
			printf("after %s %d create queue; QUEUE COUNT: %d\n", label, y, queue_count);
		
			if (*ciErrNum != CL_SUCCESS){
				printf (" Error %i in clCreateCommandQueue call: %s !!!\n\n", *ciErrNum, GetErrorCode(*ciErrNum));
				return false;
			}
	    }
		
		/*
		*queue = (cl_command_queue *) &new_queue;
		*devices = (cl_device_id *) &new_devices;
		*/
		/*
		*queue = (magma_queue_t *) new magma_queue_t[n_device];
		*devices = (cl_device_id *) new cl_device_id[n_device];
		memcpy(*queue, &new_queue, sizeof(cl_command_queue)*n_device);
		memcpy(*devices, &new_devices, sizeof(cl_device_id)*n_device);
		delete [] new_queue;
		delete [] new_devices;*/
		
	  }
	  
	}else{
	  printf(" Error %i in clGetDeviceIDs (%s) call !!!\n\n", *ciErrNum, label);
	  return false;
	  
	}
	
	
     }
     
  
  }
  return true;
}

//init platform, returns the index reference to cl_platform element that will be stored in cpPlatforms class vector
cl_int CL_MAGMA_RT::initPlatform(const cl_platform_id src_platform){
  
  cl_platform platform;
  uint index = 0;
  std::vector<cl_platform>::iterator it;
  cl_int ciErrNum = 0;
  
  if(!cpPlatforms.empty()){
    cl_platform current;
    for(cl_int i = 0; i < cpPlatforms.size(); i++){
		current = cpPlatforms.at(i);
		if(current.platform == src_platform){
			printf("Platform %d already stored. Nothing to do.\n", i);
			return i;
			
		}
    }
  }
  
  platform.platform = src_platform;
 
  
  //init GPUS
  
  printf("OpenCL GPU Device Info:\n");  
  
  if(!(initDevices(src_platform, &platform.gpu_devices, &platform.gpu_context, &platform.n_gpu, &platform.gpu_queue, CL_DEVICE_TYPE_GPU, MagmaMaxGPUs, &ciErrNum, (char *) "GPU"))){
    printf("Error! initDevices failed: %d\n", ciErrNum);
    return MAGMA_ERR_UNKNOWN;
  }
  
  //init CPUS
  printf("OpenCL CPU Device Info:\n");
  
  //NOTE #define MagmaMaxDEVs MagmaMaxGPUs (magma_types.h)
  if(!(initDevices(src_platform, &platform.cpu_devices, &platform.cpu_context, &platform.n_cpu, &platform.cpu_queue, CL_DEVICE_TYPE_CPU, MagmaMaxDEVs, &ciErrNum, (char *) "CPU"))){

    printf("Error! initDevices failed: %d\n", ciErrNum);
    return MAGMA_ERR_UNKNOWN;
  }
  
  //push platform into global vector
  cpPlatforms.push_back(platform);
  
  // iterator to vector element:
  it = std::find_if(cpPlatforms.begin(),cpPlatforms.end(),FindPlatformID(src_platform));

  if(it != cpPlatforms.end()) {
    index = it - cpPlatforms.begin();
  }
  
  return index;
}

uint CL_MAGMA_RT::GetPlatformIndex()
{  
  uint index = 0;
  std::vector<cl_platform>::iterator it;
  it = std::find_if(cpPlatforms.begin(),cpPlatforms.end(),FindPlatformID(cpPlatform));

  if(it != cpPlatforms.end()) {
    index = it - cpPlatforms.begin();
  }
  
  return index;

}


/*
 * read source code from filename
 * from Rick's clutil
 */
string CL_MAGMA_RT::fileToString(const char* filename)
{
	ifstream fileStream(filename, ios::binary | ios::in | ios::ate);

	if(fileStream.is_open() == true)
	{
		size_t fileSize = fileStream.tellg();

		char* cbuffer = new char[fileSize + 1];

		fileStream.seekg(0, ios::beg);
		fileStream.read(cbuffer, fileSize);
		cbuffer[fileSize] = '\0';

		string  memoryBuffer(cbuffer);
		delete [] cbuffer;
		return memoryBuffer;
	}
	else
	{
		printf ("Error could not open %s\n", filename);
		return NULL;
	}
}


const char* CL_MAGMA_RT::GetErrorCode(cl_int err)
{
		switch(err)
		{
			case CL_SUCCESS:
				return "No Error.";
			case CL_INVALID_MEM_OBJECT:
				return "Invalid memory object.";
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
				return "Invalid image format descriptor.";
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:
				return "Image format not supported.";
			case CL_INVALID_IMAGE_SIZE:
				return "Invalid image size.";
			case CL_INVALID_ARG_INDEX:
				return "Invalid argument index for this kernel.";
			case CL_INVALID_ARG_VALUE:
				return "Invalid argument value.";
			case CL_INVALID_SAMPLER:
				return "Invalid sampler.";
			case CL_INVALID_ARG_SIZE:
				return "Invalid argument size.";
			case CL_INVALID_BUFFER_SIZE:
				return "Invalid buffer size.";
			case CL_INVALID_HOST_PTR:
				return "Invalid host pointer.";
			case CL_INVALID_DEVICE:
				return "Invalid device.";
			case CL_INVALID_VALUE:
				return "Invalid value.";
			case CL_INVALID_CONTEXT:
				return "Invalid Context.";
			case CL_INVALID_KERNEL:
				return "Invalid kernel.";
			case CL_INVALID_PROGRAM:
				return "Invalid program object.";
			case CL_INVALID_BINARY:
				return "Invalid program binary.";
			case CL_INVALID_OPERATION:
				return "Invalid operation.";
			case CL_INVALID_BUILD_OPTIONS:
				return "Invalid build options.";
			case CL_INVALID_PROGRAM_EXECUTABLE:
				return "Invalid executable.";
			case CL_INVALID_COMMAND_QUEUE:
				return "Invalid command queue.";
			case CL_INVALID_KERNEL_ARGS:
				return "Invalid kernel arguments.";
			case CL_INVALID_WORK_DIMENSION:
				return "Invalid work dimension.";
			case CL_INVALID_WORK_GROUP_SIZE:
				return "Invalid work group size.";
			case CL_INVALID_WORK_ITEM_SIZE:
				return "Invalid work item size.";
			case CL_INVALID_GLOBAL_OFFSET:
				return "Invalid global offset (should be NULL).";
			case CL_OUT_OF_RESOURCES:
				return "Insufficient resources.";
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				return "Could not allocate mem object.";
			case CL_INVALID_EVENT_WAIT_LIST:
				return "Invalid event wait list.";
			case CL_OUT_OF_HOST_MEMORY:
				return "Out of memory on host.";
			case CL_INVALID_KERNEL_NAME:
				return "Invalid kernel name.";
			case CL_INVALID_KERNEL_DEFINITION:
				return "Invalid kernel definition.";
			case CL_BUILD_PROGRAM_FAILURE:
				return "Failed to build program.";
			case CL_MAP_FAILURE:
				return "Failed to map buffer/image";
			case -1001: //This is CL_PLATFORM_NOT_FOUND_KHR
				return "No platforms found. (Did you put ICD files in /etc/OpenCL?)";
			default:
				return "Unknown error.";
		}
}

bool CL_MAGMA_RT::Quit()
{
  
	cl_platform current;
	
	if (!HasBeenInitialized)
		return false;

	// Cleanup allocated objects
	if (commandQueue)	commandQueue = NULL;
	if(cxGPUContext)	clReleaseContext(cxGPUContext);
	
	if(ceEvent)		clReleaseEvent(ceEvent);  
	if(ckKernel)		clReleaseKernel(ckKernel);  
	
	if(cdDevices)		cdDevices = NULL;
	
	ciDeviceCount = 0;
	platform_id = 0;
	dev_type = 0;
	ceEvent = NULL;
	ckKernel = NULL;

	cpPlatform = NULL;
	cxGPUContext = NULL;
	
	if(cpPlatforms.size() > 0) {
	  
	  cpPlatforms.clear();
	  
	}
	
	
	
	

	HasBeenInitialized = false;

	return true;
}

bool CL_MAGMA_RT::Init(cl_platform_id gPlatform, cl_context gContext)
{
  if (HasBeenInitialized)
    {
      printf ("Error: CL_MAGMA_RT has been initialized\n");
      return false;
    }

  printf ("Initializing clMAGMA runtime ...\n");
  
  cl_int ciErrNum = CL_SUCCESS;

  // set the platform
  cpPlatform    = gPlatform;

  ciErrNum  = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
  cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id));
  ciErrNum |= clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);

  // set the context
  cxGPUContext = gContext;

  // create command-queues                                                                           
  commandQueue = new cl_command_queue[QUEUE_COUNT];
  for(unsigned int i = 0; i < QUEUE_COUNT; i++)
    {
      // create command queue                                                                    
      commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[0], 
					     CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
      if (ciErrNum != CL_SUCCESS)
	{
	  printf (" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
	  return false;
	}
    }

  // setup kernel name -> file name (this will be done later automatically)
  // get directory from environment variable or use default
  const char* dirstr = getenv( "MAGMA_CL_DIR" );
  if ( dirstr == NULL || strlen(dirstr) == 0 ) {
  	  dirstr = "/usr/local/magma/cl";
  	  printf( "using default MAGMA_CL_DIR = %s\n", dirstr );
  }
  // make sure it ends in /
  string dir = dirstr;
  if ( dir.size() > 0 && dir[dir.size()-1] != '/' ) {
  	  dir += '/';
  }
  Kernel2FileNamePool["sinplace_T_even_kernel"] = dir + "sinplace_transpose.co";
  Kernel2FileNamePool["sinplace_T_odd_kernel" ] = dir + "sinplace_transpose.co";
  Kernel2FileNamePool["stranspose3_32"        ] = dir + "stranspose-v2.co";
  Kernel2FileNamePool["stranspose_32"         ] = dir + "stranspose.co";
  Kernel2FileNamePool["myslaswp2"             ] = dir + "spermute-v2.co";

  Kernel2FileNamePool["dinplace_T_even_kernel"] = dir + "dinplace_transpose.co";
  Kernel2FileNamePool["dinplace_T_odd_kernel" ] = dir + "dinplace_transpose.co";
  Kernel2FileNamePool["dtranspose3_32"        ] = dir + "dtranspose-v2.co";
  Kernel2FileNamePool["dtranspose_32"         ] = dir + "dtranspose.co";
  Kernel2FileNamePool["mydlaswp2"             ] = dir + "dpermute-v2.co";

  Kernel2FileNamePool["cinplace_T_even_kernel"] = dir + "cinplace_transpose.co";
  Kernel2FileNamePool["cinplace_T_odd_kernel" ] = dir + "cinplace_transpose.co";
  Kernel2FileNamePool["ctranspose3_32"        ] = dir + "ctranspose-v2.co";
  Kernel2FileNamePool["ctranspose_32"         ] = dir + "ctranspose.co";
  Kernel2FileNamePool["myclaswp2"             ] = dir + "cpermute-v2.co";

  Kernel2FileNamePool["zinplace_T_even_kernel"] = dir + "zinplace_transpose.co";
  Kernel2FileNamePool["zinplace_T_odd_kernel" ] = dir + "zinplace_transpose.co";
  Kernel2FileNamePool["ztranspose3_32"        ] = dir + "ztranspose-v2.co";
  Kernel2FileNamePool["ztranspose_32"         ] = dir + "ztranspose.co";
  Kernel2FileNamePool["myzlaswp2"             ] = dir + "zpermute-v2.co";

//auxiliary functions
  Kernel2FileNamePool["sset_nbxnb_to_zero"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dset_nbxnb_to_zero"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["cset_nbxnb_to_zero"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zset_nbxnb_to_zero"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset_lower"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset_lower"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset_lower"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset_lower"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset_upper"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset_upper"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset_upper"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset_upper"    ] = dir + "zauxiliary.co";

//zlacpy functions
  Kernel2FileNamePool["slacpy_kernel"    ] = dir + "slacpy.co";
  Kernel2FileNamePool["dlacpy_kernel"    ] = dir + "dlacpy.co";
  Kernel2FileNamePool["clacpy_kernel"    ] = dir + "clacpy.co";
  Kernel2FileNamePool["zlacpy_kernel"    ] = dir + "zlacpy.co";

//zswap functions
  Kernel2FileNamePool["magmagpu_sswap"    ] = dir + "sswap.co";
  Kernel2FileNamePool["magmagpu_dswap"    ] = dir + "dswap.co";
  Kernel2FileNamePool["magmagpu_cswap"    ] = dir + "cswap.co";
  Kernel2FileNamePool["magmagpu_zswap"    ] = dir + "zswap.co";

  HasBeenInitialized = true;

  BuildFromBinaries( (dir + "sinplace_transpose.co").c_str() );
  BuildFromBinaries( (dir + "stranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "stranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "spermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "dinplace_transpose.co").c_str() );
  BuildFromBinaries( (dir + "dtranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "dtranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "dpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "cinplace_transpose.co").c_str() );
  BuildFromBinaries( (dir + "ctranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "ctranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "cpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "zinplace_transpose.co").c_str() );
  BuildFromBinaries( (dir + "ztranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "ztranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "zpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "sauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "dauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "cauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "zauxiliary.co"       ).c_str() );
 
  BuildFromBinaries( (dir + "slacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "dlacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "clacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "zlacpy.co"       ).c_str() );

  BuildFromBinaries( (dir + "sswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "dswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "cswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "zswap.co"       ).c_str() );

  bool rtr;
  rtr = CreateKernel("sinplace_T_even_kernel");
  if (rtr==false)
    printf ("error creating kernel sinplace_T_even_kernel\n");
  rtr = CreateKernel("sinplace_T_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel sinplace_T_odd_kernel\n");
  rtr = CreateKernel("stranspose3_32");
  if (rtr==false)
    printf ("error creating kernel stranspose3_32\n");
  rtr = CreateKernel("stranspose_32");
  if (rtr==false)
    printf ("error creating kernel stranspose_32\n");
  rtr = CreateKernel("myslaswp2");
  if (rtr==false)
    printf ("error creating kernel myslaswp2\n");

  rtr = CreateKernel("dinplace_T_even_kernel");
  if (rtr==false)
    printf ("error creating kernel dinplace_T_even_kernel\n");
  rtr = CreateKernel("dinplace_T_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel dinplace_T_odd_kernel\n");
  rtr = CreateKernel("dtranspose3_32");
  if (rtr==false)
    printf ("error creating kernel dtranspose3_32\n");
  rtr = CreateKernel("dtranspose_32");
  if (rtr==false)
    printf ("error creating kernel dtranspose_32\n");
  rtr = CreateKernel("mydlaswp2");
  if (rtr==false)
    printf ("error creating kernel mydlaswp2\n");

  rtr = CreateKernel("cinplace_T_even_kernel");
  if (rtr==false)
    printf ("error creating kernel cinplace_T_even_kernel\n");
  rtr = CreateKernel("cinplace_T_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel cinplace_T_odd_kernel\n");
  rtr = CreateKernel("ctranspose3_32");
  if (rtr==false)
    printf ("error creating kernel ctranspose3_32\n");
  rtr = CreateKernel("ctranspose_32");
  if (rtr==false)
    printf ("error creating kernel ctranspose_32\n");
  rtr = CreateKernel("myclaswp2");
  if (rtr==false)
    printf ("error creating kernel myclaswp2\n");

  rtr = CreateKernel("zinplace_T_even_kernel");
  if (rtr==false)
    printf ("error creating kernel zinplace_T_even_kernel\n");
  rtr = CreateKernel("zinplace_T_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel zinplace_T_odd_kernel\n");
  rtr = CreateKernel("ztranspose3_32");
  if (rtr==false)
    printf ("error creating kernel ztranspose3_32\n");
  rtr = CreateKernel("ztranspose_32");
  if (rtr==false)
    printf ("error creating kernel ztranspose_32\n");
  rtr = CreateKernel("myzlaswp2");
  if (rtr==false)
    printf ("error creating kernel myzlaswp2\n");

  rtr = CreateKernel("sset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel sset_nbxnb_zero\n");
  rtr = CreateKernel("dset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel dset_nbxnb_zero\n");
  rtr = CreateKernel("cset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel cset_nbxnb_zero\n");
  rtr = CreateKernel("zset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel zset_nbxnb_zero\n");
  rtr = CreateKernel("slaset");
  if (rtr==false)
    printf ("error creating kernel slaset\n");
  rtr = CreateKernel("dlaset");
  if (rtr==false)
    printf ("error creating kernel dlaset\n");
  rtr = CreateKernel("claset");
  if (rtr==false)
    printf ("error creating kernel claset");
  rtr = CreateKernel("zlaset");
  if (rtr==false)
    printf ("error creating kernel zlaset\n");
  rtr = CreateKernel("slaset_lower");
  if (rtr==false)
    printf ("error creating kernel slaset_lower\n");
  rtr = CreateKernel("dlaset_lower");
  if (rtr==false)
    printf ("error creating kernel dlaset_lower\n");
  rtr = CreateKernel("claset_lower");
  if (rtr==false)
    printf ("error creating kernel claset_lower");
  rtr = CreateKernel("zlaset_lower");
  if (rtr==false)
    printf ("error creating kernel zlaset_lower\n");
  rtr = CreateKernel("slaset_upper");
  if (rtr==false)
    printf ("error creating kernel slaset_upper\n");
  rtr = CreateKernel("dlaset_upper");
  if (rtr==false)
    printf ("error creating kernel dlaset_upper\n");
  rtr = CreateKernel("claset_upper");
  if (rtr==false)
    printf ("error creating kernel claset_upper");
  rtr = CreateKernel("zlaset_upper");
  if (rtr==false)
    printf ("error creating kernel zlaset_upper\n");
 
  rtr = CreateKernel("slacpy_kernel");
  if (rtr==false)
	  printf ("error creating kernel slacpy_kernel\n");
  rtr = CreateKernel("dlacpy_kernel");
  if (rtr==false)
	  printf ("error creating kernel dlacpy_kernel\n");
  rtr = CreateKernel("clacpy_kernel");
  if (rtr==false)
	  printf ("error creating kernel clacpy_kernel");
  rtr = CreateKernel("zlacpy_kernel");
  if (rtr==false)
	  printf ("error creating kernel zlacpy_kernel\n");

  rtr = CreateKernel("magmagpu_sswap");
  if (rtr==false)
	  printf ("error creating kernel magmagpu_sswap\n");
  rtr = CreateKernel("magmagpu_dswap");
  if (rtr==false)
	  printf ("error creating kernel magmagpu_dswap\n");
  rtr = CreateKernel("magmagpu_cswap");
  if (rtr==false)
	  printf ("error creating kernel magmagpu_cswap\n");
  rtr = CreateKernel("magmagpu_zswap");
  if (rtr==false)
	  printf ("error creating kernel magmagpu_zswap\n");

  return true;
}

//BELGIOVI

bool CL_MAGMA_RT::generateKernelMap(){	//Genera la mappa dei Kernel per la piattaforma OpenCL attuale. Restituisce il numero di kernel inizializzati
  
    
  string strTypeDir;
  char buf[5];

  // setup kernel name -> file name (this will be done later automatically)
  // get directory from environment variable or use default
  const char* dirstr = getenv( "MAGMA_CL_DIR" );
  if ( dirstr == NULL || strlen(dirstr) == 0 ) {
  	  dirstr = "/usr/local/magma/cl";
  	  printf( "using default MAGMA_CL_DIR = %s\n", dirstr );
  }
  
  // make sure it ends in /
  string dir = dirstr;
  if ( dir.size() > 0 && dir[dir.size()-1] != '/' ) {
  	  dir += '/';
  }
  switch(dev_type){
    case CL_DEVICE_TYPE_CPU:
	strTypeDir = "CL_DEVICE_TYPE_CPU";
	break;
    default:
	strTypeDir = "CL_DEVICE_TYPE_GPU";
  }
  
  sprintf(buf, "%u",platform_id); //platform_id usato per la cartella che contiene i .co
  
  dir += string("cl_build/") + string(buf) + '/' + strTypeDir + '/';
  
  Kernel2FileNamePool["sinplace_T_even_kernel"] = dir + "sinplace_transpose.co";
  Kernel2FileNamePool["sinplace_T_odd_kernel" ] = dir + "sinplace_transpose.co";
  Kernel2FileNamePool["stranspose3_32"        ] = dir + "stranspose-v2.co";
  Kernel2FileNamePool["stranspose_32"         ] = dir + "stranspose.co";
  Kernel2FileNamePool["myslaswp2"             ] = dir + "spermute-v2.co";

  Kernel2FileNamePool["dinplace_T_even_kernel"] = dir + "dinplace_transpose.co";
  Kernel2FileNamePool["dinplace_T_odd_kernel" ] = dir + "dinplace_transpose.co";
  Kernel2FileNamePool["dtranspose3_32"        ] = dir + "dtranspose-v2.co";
  Kernel2FileNamePool["dtranspose_32"         ] = dir + "dtranspose.co";
  Kernel2FileNamePool["mydlaswp2"             ] = dir + "dpermute-v2.co";

  Kernel2FileNamePool["cinplace_T_even_kernel"] = dir + "cinplace_transpose.co";
  Kernel2FileNamePool["cinplace_T_odd_kernel" ] = dir + "cinplace_transpose.co";
  Kernel2FileNamePool["ctranspose3_32"        ] = dir + "ctranspose-v2.co";
  Kernel2FileNamePool["ctranspose_32"         ] = dir + "ctranspose.co";
  Kernel2FileNamePool["myclaswp2"             ] = dir + "cpermute-v2.co";

  Kernel2FileNamePool["zinplace_T_even_kernel"] = dir + "zinplace_transpose.co";
  Kernel2FileNamePool["zinplace_T_odd_kernel" ] = dir + "zinplace_transpose.co";
  Kernel2FileNamePool["ztranspose3_32"        ] = dir + "ztranspose-v2.co";
  Kernel2FileNamePool["ztranspose_32"         ] = dir + "ztranspose.co";
  Kernel2FileNamePool["myzlaswp2"             ] = dir + "zpermute-v2.co";

//auxiliary functions
  Kernel2FileNamePool["sset_nbxnb_to_zero"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dset_nbxnb_to_zero"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["cset_nbxnb_to_zero"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zset_nbxnb_to_zero"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset_lower"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset_lower"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset_lower"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset_lower"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset_upper"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset_upper"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset_upper"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset_upper"    ] = dir + "zauxiliary.co";

//zlacpy functions
  Kernel2FileNamePool["slacpy_kernel"    ] = dir + "slacpy.co";
  Kernel2FileNamePool["dlacpy_kernel"    ] = dir + "dlacpy.co";
  Kernel2FileNamePool["clacpy_kernel"    ] = dir + "clacpy.co";
  Kernel2FileNamePool["zlacpy_kernel"    ] = dir + "zlacpy.co";

//zswap functions
  Kernel2FileNamePool["magmagpu_sswap"    ] = dir + "sswap.co";
  Kernel2FileNamePool["magmagpu_dswap"    ] = dir + "dswap.co";
  Kernel2FileNamePool["magmagpu_cswap"    ] = dir + "cswap.co";
  Kernel2FileNamePool["magmagpu_zswap"    ] = dir + "zswap.co";

  BuildFromBinaries( (dir + "sinplace_transpose.co").c_str() );
  BuildFromBinaries( (dir + "stranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "stranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "spermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "dinplace_transpose.co").c_str() );
  BuildFromBinaries( (dir + "dtranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "dtranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "dpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "cinplace_transpose.co").c_str() );
  BuildFromBinaries( (dir + "ctranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "ctranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "cpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "zinplace_transpose.co").c_str() );
  BuildFromBinaries( (dir + "ztranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "ztranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "zpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "sauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "dauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "cauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "zauxiliary.co"       ).c_str() );

  BuildFromBinaries( (dir + "slacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "dlacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "clacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "zlacpy.co"       ).c_str() );

  BuildFromBinaries( (dir + "sswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "dswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "cswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "zswap.co"       ).c_str() );

  bool rtr;
  rtr = CreateKernel("sinplace_T_even_kernel");
  if (rtr==false)
    printf ("error creating kernel sinplace_T_even_kernel\n");
  rtr = CreateKernel("sinplace_T_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel sinplace_T_odd_kernel\n");
  rtr = CreateKernel("stranspose3_32");
  if (rtr==false)
    printf ("error creating kernel stranspose3_32\n");
  rtr = CreateKernel("stranspose_32");
  if (rtr==false)
    printf ("error creating kernel stranspose_32\n");
  rtr = CreateKernel("myslaswp2");
  if (rtr==false)
    printf ("error creating kernel myslaswp2\n");

  rtr = CreateKernel("dinplace_T_even_kernel");
  if (rtr==false)
    printf ("error creating kernel dinplace_T_even_kernel\n");
  rtr = CreateKernel("dinplace_T_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel dinplace_T_odd_kernel\n");
  rtr = CreateKernel("dtranspose3_32");
  if (rtr==false)
    printf ("error creating kernel dtranspose3_32\n");
  rtr = CreateKernel("dtranspose_32");
  if (rtr==false)
    printf ("error creating kernel dtranspose_32\n");
  rtr = CreateKernel("mydlaswp2");
  if (rtr==false)
    printf ("error creating kernel mydlaswp2\n");

  rtr = CreateKernel("cinplace_T_even_kernel");
  if (rtr==false)
    printf ("error creating kernel cinplace_T_even_kernel\n");
  rtr = CreateKernel("cinplace_T_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel cinplace_T_odd_kernel\n");
  rtr = CreateKernel("ctranspose3_32");
  if (rtr==false)
    printf ("error creating kernel ctranspose3_32\n");
  rtr = CreateKernel("ctranspose_32");
  if (rtr==false)
    printf ("error creating kernel ctranspose_32\n");
  rtr = CreateKernel("myclaswp2");
  if (rtr==false)
    printf ("error creating kernel myclaswp2\n");

  rtr = CreateKernel("zinplace_T_even_kernel");
  if (rtr==false)
    printf ("error creating kernel zinplace_T_even_kernel\n");
  rtr = CreateKernel("zinplace_T_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel zinplace_T_odd_kernel\n");
  rtr = CreateKernel("ztranspose3_32");
  if (rtr==false)
    printf ("error creating kernel ztranspose3_32\n");
  rtr = CreateKernel("ztranspose_32");
  if (rtr==false)
    printf ("error creating kernel ztranspose_32\n");
  rtr = CreateKernel("myzlaswp2");
  if (rtr==false)
    printf ("error creating kernel myzlaswp2\n");

  rtr = CreateKernel("sset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel sset_nbxnb_zero\n");
  rtr = CreateKernel("dset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel dset_nbxnb_zero\n");
  rtr = CreateKernel("cset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel cset_nbxnb_zero\n");
  rtr = CreateKernel("zset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel zset_nbxnb_zero\n");
  rtr = CreateKernel("slaset");
  if (rtr==false)
    printf ("error creating kernel slaset\n");
  rtr = CreateKernel("dlaset");
  if (rtr==false)
    printf ("error creating kernel dlaset\n");
  rtr = CreateKernel("claset");
  if (rtr==false)
    printf ("error creating kernel claset");
  rtr = CreateKernel("zlaset");
  if (rtr==false)
    printf ("error creating kernel zlaset\n");
  rtr = CreateKernel("slaset_lower");
  if (rtr==false)
    printf ("error creating kernel slaset_lower\n");
  rtr = CreateKernel("dlaset_lower");
  if (rtr==false)
    printf ("error creating kernel dlaset_lower\n");
  rtr = CreateKernel("claset_lower");
  if (rtr==false)
    printf ("error creating kernel claset_lower");
  rtr = CreateKernel("zlaset_lower");
  if (rtr==false)
    printf ("error creating kernel zlaset_lower\n");
  rtr = CreateKernel("slaset_upper");
  if (rtr==false)
    printf ("error creating kernel slaset_upper\n");
  rtr = CreateKernel("dlaset_upper");
  if (rtr==false)
    printf ("error creating kernel dlaset_upper\n");
  rtr = CreateKernel("claset_upper");
  if (rtr==false)
    printf ("error creating kernel claset_upper");
  rtr = CreateKernel("zlaset_upper");
  if (rtr==false)
    printf ("error creating kernel zlaset_upper\n");

  rtr = CreateKernel("slacpy_kernel");
  if (rtr==false)
	  printf ("error creating kernel slacpy_kernel\n");
  rtr = CreateKernel("dlacpy_kernel");
  if (rtr==false)
	  printf ("error creating kernel dlacpy_kernel\n");
  rtr = CreateKernel("clacpy_kernel");
  if (rtr==false)
	  printf ("error creating kernel clacpy_kernel");
  rtr = CreateKernel("zlacpy_kernel");
  if (rtr==false)
	  printf ("error creating kernel zlacpy_kernel\n");

  rtr = CreateKernel("magmagpu_sswap");
  if (rtr==false)
	  printf ("error creating kernel magmagpu_sswap\n");
  rtr = CreateKernel("magmagpu_dswap");
  if (rtr==false)
	  printf ("error creating kernel magmagpu_dswap\n");
  rtr = CreateKernel("magmagpu_cswap");
  if (rtr==false)
	  printf ("error creating kernel magmagpu_cswap\n");
  rtr = CreateKernel("magmagpu_zswap");
  if (rtr==false)
	  printf ("error creating kernel magmagpu_zswap\n");
  
  BigKernelPool[platform_id][dev_type] = KernelPool;
  
  return true;
}

cl_kernel CL_MAGMA_RT::getKernel(const string kernel_name)
{
  kernel_map &elMap = BigKernelPool[platform_id][dev_type];
  kernel_map::iterator it;
  
  it = elMap.find(kernel_name);
  if (it == elMap.end()){
    return NULL;
  }
  
  return it->second;
  
}


void CL_MAGMA_RT::PrintKernelMap(){
  //TEST scrorrimento
  platform_map &outerMap = BigKernelPool;
  //std::cout << "CL_DEVICE_TYPE_CPU: " << CL_DEVICE_TYPE_CPU << ", CL_DEVICE_TYPE_GPU: " << CL_DEVICE_TYPE_GPU << std::endl;
  for (platform_map::iterator i = outerMap.begin(), iend = outerMap.end(); i != iend; ++i)
  {
    devType_map &innerMap = i->second;
    
    for (devType_map::iterator j = innerMap.begin(), jend = innerMap.end(); j != jend; ++j)
    {
	kernel_map &elMap = j->second;
	
	for (kernel_map::iterator k = elMap.begin(), kend = elMap.end(); k != kend; ++k)
	{  
	  std::cout << i->first << " : " << j->first << " : "<< k->first << std::endl;
	}
    }
  }
}

bool CL_MAGMA_RT::InitAll() 
{
  if (HasBeenInitialized)
    {
      printf ("Error: CL_MAGMA_RT has been initialized\n");
      return false;
    }

  printf ("Initializing clMAGMA runtime ...\n\n");
  
  cl_int ciErrNum = CL_SUCCESS;
  char chBuffer[1024];
  cl_platform_id* clPlatformIDs;
  cl_uint n_platform;
  cl_int index = 0;
  
  // Init StarPU
  /*
  int err = starpu_init(NULL);
  if( err == 0 ){
    printf("Error: starpu_init returned %d\n", err);
    return false;
  }
  */
  
  // set the platform
  ciErrNum = clGetPlatformIDs (0, NULL, &n_platform);
  
  if (ciErrNum != CL_SUCCESS){
    printf(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
  }else if(n_platform == 0){
    printf("No OpenCL platform found!\n\n");
    return false;
  }else if((clPlatformIDs = new cl_platform_id[n_platform]) != NULL){
    printf("-> Found %d OpenCL platform\n\n", n_platform);

    ciErrNum = clGetPlatformIDs (n_platform, clPlatformIDs, NULL);
    
    for(cl_uint i = 0; i < n_platform; i++){
      
      ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
      if(ciErrNum == MAGMA_SUCCESS){
	printf("[%d] %s\t",i, chBuffer);
      }else{
	printf("\t Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
	return false;
      }
      ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_VERSION, 1024, &chBuffer, NULL);
      if (ciErrNum != MAGMA_SUCCESS){
	printf("\t Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
	return false;
      } else {
	  printf(" CL_PLATFORM_VERSION: \t%s\n\n", chBuffer);
	  // Get and log OpenCL device info 
	  // save platform 
	  index = initPlatform(clPlatformIDs[i]);
	  
	  if(index < 0){
	    printf("Error during initPlatform(): %d\n", index);
	    return false;
	  }
      }
    }// end platform for()
    delete [] clPlatformIDs;
    
    
  }else {
    printf("Failed to allocate memory for cl_platform ID's!\n\n");
    return false;
  }
   
  // set the default platform and context
  /*cl_platform platform;
  if(cpPlatforms.size() > 0){
    platform = cpPlatforms.at(0);
    cpPlatform    = platform.platform;
    cxGPUContext  = platform.gpu_context;
    commandQueue  = platform.gpu_queue;
    cdDevices = platform.gpu_devices;
    //TODO: controllare sollevamento eccezione std::bad_alloc per ciDeviceCount > 1 
    ciDeviceCount = platform.n_gpu;
    dev_type = CL_DEVICE_TYPE_GPU;
    platform_id = 0;

  }else{
    printf("No OpenCL platforms found!");
    return false;
  }*/
  
  if(GetNumPlatform() == 0){
    printf("Error setting the default platform!");
    return false;
  }
   
  HasBeenInitialized = true;
  
  for(uint i = 0; i < GetNumPlatform(); i++){	//for(i..numPlatform): PER OGNI PIATTAFORMA (E OGNI TIPO DI DEVICE) GENERIAMO LA MAPPA DEI KERNEL
    
    /* NOTE:
     * . setPlatform() GPU
     * . funzione generateKernelMap()
     * . setPlatform() CPU
     * . funzione generateKernelMap()
     */
    if(SetPlatform(i, CL_DEVICE_TYPE_GPU) != NULL){
      generateKernelMap();
    }
    
    if(SetPlatform(i, CL_DEVICE_TYPE_CPU) != NULL){
      generateKernelMap();
    }
    
  }//end for(i..numPlatform)
  
  //Impostiamo la GPU (o CPU) per il device acceleratore di default
  if(SetPlatform(0, CL_DEVICE_TYPE_GPU) == NULL){
    if(SetPlatform(0, CL_DEVICE_TYPE_CPU) == NULL){
      printf("Error setting the default platform!");
      return false;
    }  
  }
  
  
  
  return true;
}


bool CL_MAGMA_RT::Init()
{
	if (HasBeenInitialized)
	{
		printf ("Error: CL_MAGMA_RT has been initialized\n");
		return false;
	}

	printf ("Initializing...\n");

	/*
	 * initialize OpenCL runtime
	 */
	cl_int ciErrNum = CL_SUCCESS;
	
	// Get the platform
	/*cl_uint ione = 1;
	ciErrNum = clGetPlatformIDs(1, &cpPlatform, &ione);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: Failed to create OpenCL context!\n");
		return ciErrNum;
	}

	ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
	cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id));
	ciErrNum |= clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: clGetDeviceIDs at %d in file %s!\n", __LINE__, __FILE__);
		return false;
	}

	//Create the context
	cxGPUContext = clCreateContext(0, ciDeviceCount, cdDevices, NULL, NULL, &ciErrNum);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: Failed to create OpenCL context!\n");
		return false;
	}   
	*/
	
	/*
	// Find out how many GPU's to compute on all available GPUs
	size_t nDeviceBytes;
	ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
	if (ciErrNum != CL_SUCCESS)
	{   
		printf (" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
		return ciErrNum;
	}
	else if (ciDeviceCount == 0)
	{
		printf (" There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
		return false;
	}
	ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id); 
	

	// show device 
	for(unsigned int i = 0; i < ciDeviceCount; i++)
	{
		// get and print the device for this queue
		//cl_device_id device = oclGetDev(cxGPUContext, i);

		char deviceName[1024];
		memset(deviceName, '\0', 1024);
		clGetDeviceInfo(cdDevices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
		printf ("Device: %s\n", deviceName);
	}

	// create command-queues
	commandQueue = new cl_command_queue[QUEUE_COUNT];
	for(unsigned int i = 0; i < QUEUE_COUNT; i++)
	{
		// create command queue
		commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[0], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
		if (ciErrNum != CL_SUCCESS)
		{
			printf (" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
			return false;
		}
	}*/
	char chBuffer[1024];
	cl_platform_id* clPlatformIDs;
	cl_uint n_platform;
	cl_int index = 0;
	
	ciErrNum = clGetPlatformIDs (0, NULL, &n_platform);
  
	if (ciErrNum != CL_SUCCESS){
	  printf(" Error %i in clGetPlatformIDs Call (%s)!!!\n\n", ciErrNum, GetErrorCode(ciErrNum));
	}else if(n_platform == 0){
	  printf("No OpenCL platform found!\n\n");
	  return false;
	}else if((clPlatformIDs = new cl_platform_id[n_platform]) != NULL){
	  printf("-> Found %d OpenCL platform\n\n", n_platform);

	  ciErrNum = clGetPlatformIDs (n_platform, clPlatformIDs, NULL);
    
	  for(cl_uint i = 0; i < n_platform; i++){
      
	    ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
	    if(ciErrNum == MAGMA_SUCCESS){
	      printf("[%d] %s\t",i, chBuffer);
	    }else{
	      printf("\t Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
	      return false;
	    }
	  ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_VERSION, 1024, &chBuffer, NULL);
	  if (ciErrNum != MAGMA_SUCCESS){
	    printf("\t Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
	    return false;
	  } else {
	      printf(" CL_PLATFORM_VERSION: \t%s\n\n", chBuffer);
	      // Get and log OpenCL device info 
	      // save platform 
	      index = initPlatform(clPlatformIDs[i]);
	  
	      if(index < 0){
		printf("Error during initPlatform(): %d\n", index);
		return false;
	      }
	    }
	  }// end platform for()
	  delete [] clPlatformIDs;
	}else {
	  printf("Failed to allocate memory for cl_platform ID's!\n\n");
	  return false;
	}
	
	// setup kernel name -> file name (this will be done later automatically)
    /*string dir = "/Users/mgates/Documents/magma-cl/interface_opencl/";
	Kernel2FileNamePool["sinplace_T_even_kernel"] = dir + string("sinplace_transpose.co");
	Kernel2FileNamePool["sinplace_T_odd_kernel" ] = dir + string("sinplace_transpose.co");
	Kernel2FileNamePool["stranspose3_32"        ] = dir + string("stranspose-v2.co");
	Kernel2FileNamePool["stranspose_32"         ] = dir + string("stranspose.co");
	Kernel2FileNamePool["myslaswp2"             ] = dir + string("spermute-v2.co");

	Kernel2FileNamePool["dinplace_T_even_kernel"] = dir + string("dinplace_transpose.co");
	Kernel2FileNamePool["dinplace_T_odd_kernel" ] = dir + string("dinplace_transpose.co");
	Kernel2FileNamePool["dtranspose3_32"        ] = dir + string("dtranspose-v2.co");
	Kernel2FileNamePool["dtranspose_32"         ] = dir + string("dtranspose.co");
	Kernel2FileNamePool["mydlaswp2"             ] = dir + string("dpermute-v2.co");

	Kernel2FileNamePool["cinplace_T_even_kernel"] = dir + string("cinplace_transpose.co");
	Kernel2FileNamePool["cinplace_T_odd_kernel" ] = dir + string("cinplace_transpose.co");
	Kernel2FileNamePool["ctranspose3_32"        ] = dir + string("ctranspose-v2.co");
	Kernel2FileNamePool["ctranspose_32"         ] = dir + string("ctranspose.co");
	Kernel2FileNamePool["myclaswp2"             ] = dir + string("cpermute-v2.co");

	Kernel2FileNamePool["zinplace_T_even_kernel"] = dir + string("zinplace_transpose.co");
	Kernel2FileNamePool["zinplace_T_odd_kernel" ] = dir + string("zinplace_transpose.co");
	Kernel2FileNamePool["ztranspose3_32"        ] = dir + string("ztranspose-v2.co");
	Kernel2FileNamePool["ztranspose_32"         ] = dir + string("ztranspose.co");
	Kernel2FileNamePool["myzlaswp2"             ] = dir + string("zpermute-v2.co");

	//auxiliary functions
	Kernel2FileNamePool["sset_nbxnb_to_zero"    ] = dir + string("sauxiliary.co");
	Kernel2FileNamePool["dset_nbxnb_to_zero"    ] = dir + string("dauxiliary.co");
	Kernel2FileNamePool["cset_nbxnb_to_zero"    ] = dir + string("cauxiliary.co");
	Kernel2FileNamePool["zset_nbxnb_to_zero"    ] = dir + string("zauxiliary.co");
	Kernel2FileNamePool["slaset"    ] = dir + string("sauxiliary.co");
	Kernel2FileNamePool["dlaset"    ] = dir + string("dauxiliary.co");
	Kernel2FileNamePool["claset"    ] = dir + string("cauxiliary.co");
	Kernel2FileNamePool["zlaset"    ] = dir + string("zauxiliary.co");
	Kernel2FileNamePool["slaset_lower"    ] = dir + string("sauxiliary.co");
	Kernel2FileNamePool["dlaset_lower"    ] = dir + string("dauxiliary.co");
	Kernel2FileNamePool["claset_lower"    ] = dir + string("cauxiliary.co");
	Kernel2FileNamePool["zlaset_lower"    ] = dir + string("zauxiliary.co");
	Kernel2FileNamePool["slaset_upper"    ] = dir + string("sauxiliary.co");
	Kernel2FileNamePool["dlaset_upper"    ] = dir + string("dauxiliary.co");
	Kernel2FileNamePool["claset_upper"    ] = dir + string("cauxiliary.co");
	Kernel2FileNamePool["zlaset_upper"    ] = dir + string("zauxiliary.co");
    
	//zlacpy functions
	Kernel2FileNamePool["slacpy_kernel"    ] = dir + string("slacpy.co");
	Kernel2FileNamePool["dlacpy_kernel"    ] = dir + string("dlacpy.co");
	Kernel2FileNamePool["clacpy_kernel"    ] = dir + string("clacpy.co");
	Kernel2FileNamePool["zlacpy_kernel"    ] = dir + string("zlacpy.co");

	//zswap functions
	Kernel2FileNamePool["magmagpu_sswap"    ] = dir + string("sswap.co");
	Kernel2FileNamePool["magmagpu_dswap"    ] = dir + string("dswap.co");
	Kernel2FileNamePool["magmagpu_cswap"    ] = dir + string("cswap.co");
	Kernel2FileNamePool["magmagpu_zswap"    ] = dir + string("zswap.co");*/

	HasBeenInitialized = true;
	return true;
}

//BELGIOVI end

int CL_MAGMA_RT::GatherFilesToCompile( const char* FileNameList, vector<string>& FileNames)
{
	if (FileNameList==NULL || strlen(FileNameList)==0)
		return -1;

	ifstream fileStream(FileNameList, ifstream::in);
	
	int num=0;
	if(fileStream.is_open())
	{
		while (!fileStream.eof())
		{
			char buff[512];

			fileStream.getline (buff,512);
			
			if (strlen(buff) && buff[0]!='#')
			{
				FileNames.push_back (string(buff));
				memset (buff, ' ', 512);
				num++;
			}
		}

	}
	fileStream.close();

	return num;
}

/*
 * this function build .cl files and store the bits to .o files
 */
bool CL_MAGMA_RT::CompileSourceFiles( const char* FileNameList )
{
	if (FileNameList==NULL)
		return false;

	//read from clfile for a list of cl files to compile  
	vector<string> FileNames;
	int NumOfFiles = GatherFilesToCompile (FileNameList, FileNames);

	if (NumOfFiles==0)
		return false;

	//compile each cl file
	vector<string>::iterator it;
	for (it=FileNames.begin(); it<FileNames.end(); it++ )
	{ 
		printf ("compiling %s\n", it->c_str());
		bool ret = CompileFile (it->c_str()); 
		if (ret==false)
		{
			printf ("Error while trying to compile %s\n", it->c_str());
			return false;
		}
	}

	return true;
}

bool CL_MAGMA_RT::CompileFile(const char *FileName)
{
	if (FileName==NULL)
	{
		printf ("Error: file name empty on line %d in %s\n", __LINE__, __FILE__);
		return false;
	}

	if (!HasBeenInitialized)
		Init();

	// read in the kernel source
	string fileStrings;

	fileStrings = fileToString(FileName);
	const char *filePointers = fileStrings.c_str();

	// Create the program
	cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&filePointers, NULL, &ciErrNum);
	if (ciErrNum != CL_SUCCESS)
	{
		printf ("Error: clCreateProgramWithSource at %d in %s\n", __LINE__, __FILE__);
		return false;
	}
	
	// Build the program
	// MSUT do this otherwise clGetProgramInfo return zeros for binary sizes
	ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
		printf ("clBuildProgram error at %d in %s\n", __LINE__, __FILE__);
		return false;
	}

	// obtain the binary
	size_t num_of_binaries=0; 
	clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_of_binaries, NULL);

	size_t *binary_sizes = new size_t[num_of_binaries];

	ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, num_of_binaries*sizeof(size_t*), binary_sizes, NULL);
	if (ciErrNum!=CL_SUCCESS)
	{
		printf ("Error: clGetProgramInfo %s at line %d, file %s\n", GetErrorCode (ciErrNum), __LINE__, __FILE__); 
		return false;
	}
	
	char **binaries = new char*[num_of_binaries];
	for (size_t i=0; i<num_of_binaries; i++)
		binaries[i] = new char[binary_sizes[i]];

	ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, (size_t)num_of_binaries*sizeof(unsigned char*), binaries, NULL);
	if (ciErrNum!=CL_SUCCESS)
	{
		// write out standard error, Build Log and PTX, then cleanup and exit
		printf ("clGetProgramInfo at %d in %s\n", __LINE__, __FILE__);
		return false;
	}

	// prepare the output file name, .cl --> .co
	string strFileName(FileName);

	size_t found;
	found=strFileName.find_last_of(".cl");
	strFileName.replace(found-1, 2, "co");
	
	//BELGIOVI
	//Inseriamo i binari in una cartella cl_build: la creiamo se non esiste
	string strBuildDirName = "cl_build";
	ciErrNum = mkdir(strBuildDirName.c_str(), S_IRWXU);
	if(ciErrNum != 0){
	  printf("cl_build creation: %s\n", strerror(errno));
	}
	string strTypeDir;
	
	switch(dev_type){
	  case CL_DEVICE_TYPE_CPU:
	    strTypeDir = "CL_DEVICE_TYPE_CPU";
	    break;
	  default:
	    strTypeDir = "CL_DEVICE_TYPE_GPU";
	}
	
	char buf[5];
	sprintf(buf, "%u",platform_id); //platform_id usato per la cartella che contiene i .co
	strBuildDirName += string("/") + string(buf);
	ciErrNum = mkdir(strBuildDirName.c_str(), S_IRWXU);
	if(ciErrNum != 0){
	  printf("%s creation: %s\n", strBuildDirName.c_str(), strerror(errno));
	}
	strBuildDirName += string("/") + strTypeDir;
	ciErrNum = mkdir(strBuildDirName.c_str(), S_IRWXU);
	if(ciErrNum != 0){
	  printf("%s creation: %s\n", strBuildDirName.c_str(), strerror(errno));
	}
	//Aggiungiamo il nome della cartella cl_build al sorgente e la cartella di tipo-device
	strFileName = strBuildDirName + string("/") + strFileName;
	//BELGIOVI end
	
	if(!fileExists(strFileName)){
	  
	  // write binaries to files
	  ofstream fileStream(strFileName.c_str(), ofstream::binary);
	  
	  if (fileStream.is_open() == true)
	  {
		  for (size_t i=0; i<num_of_binaries; i++)
		  {
			  fileStream.write ((const char *)(binary_sizes+i), (size_t)sizeof(binary_sizes[i]));
		  }
		  for (size_t i=0; i<num_of_binaries; i++)
			  fileStream.write ((const char*)binaries[i], (size_t)binary_sizes[i]);

		  fileStream.close();
	  }
	  else
	  {
		  printf ("Error: could not create binary file %s\n", strFileName.c_str());
		  return false;
	  }
	  printf("%s compiled correctly!\n", strFileName.c_str());
	  
	} else printf("%s file already exists! Nothing to do.\n", strFileName.c_str());
	


	// cleanup
	delete [] binary_sizes;
	for (size_t i=0; i<num_of_binaries; i++)
		delete [] binaries[i];
	delete [] binaries;

	return true;
}

bool CL_MAGMA_RT::BuildFromBinaries(const char *FileName)
{
	if (FileName==NULL)
	{
		printf ("Error: file name empty on line %d in %s\n", __LINE__, __FILE__);
		return false;
	}
	
	cl_uint num_of_binaries=0;
	size_t *binary_sizes;
	unsigned char **binaries;

	// load binary from file
	ifstream fileStream(FileName, ios::binary | ios::in | ios::ate);
	
	if(fileStream.is_open() == true)
	{
		//Posizione all'inizio del file
		fileStream.seekg(0, ios::beg);
		//numero file binari = numero di device (???)
		num_of_binaries = ciDeviceCount;
		//viene allocato un size_t in binary_sizes per ogni device
		binary_sizes = new size_t[num_of_binaries];
		//per ogni device...
		for (size_t i=0; i<num_of_binaries; i++)
			//scriviamo in binary_sizes+i (che sarebbe binary_sizes[i])
			// la dimensione in byte del binario che stiamo caricando
			fileStream.read((char*)(binary_sizes+i), sizeof(binary_sizes[0]));
		
		//inizializziamo il **binaries come array di unsigned char* per il num di device nel contesto
		binaries = new unsigned char*[num_of_binaries];
		for (size_t i=0; i<num_of_binaries; i++)
		{
			//allochiamo il numero di bytes necessari a contenere il binario
			binaries[i] = new unsigned char[binary_sizes[i]];
			//copiamo il contenuto del binario nella parte di memoria allocata
			fileStream.read((char*)binaries[i], (size_t)binary_sizes[i]);
			
		}
		
		fileStream.close();
		

	}
	else
	{
		printf ("Error could not open %s\n", FileName);
		return false;
	}

	// build program from binaries
	cl_int ciErrNum2 = 0;
	cl_program cpProgram = clCreateProgramWithBinary(
		cxGPUContext, num_of_binaries, cdDevices, 
		(const size_t*)binary_sizes, (const unsigned char **)binaries, &ciErrNum2, &ciErrNum);
	if (ciErrNum != CL_SUCCESS)
	{
		// write out standard error, Build Log and PTX, then cleanup and exit
		printf ("clCreateProgramWithBinary failed at %d in %s, ciErrNum2: %d, ciErrNum1: %d\n", __LINE__, __FILE__, ciErrNum2, ciErrNum);
		return false;
	}
	
	ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
		// write out standard error, Build Log and PTX, then cleanup and exit

		return false;
	}
	
	// put program in the pool
	ProgramPool[string(FileName)] = cpProgram;

	delete [] binary_sizes;
	for (size_t i=0; i<num_of_binaries; i++)
		delete [] binaries[i];
	delete [] binaries;

	return true;
}

/*
 * map kernel name to file
 * incomplete
 */
bool CL_MAGMA_RT::BuildKernelMap(const char *FileNameList)
{
	if (FileNameList==NULL)
		return false;

	/*
	//read from clfile for a list of cl files to compile  
	vector<string> FileNames;
	int NumOfFiles = GatherFilesToCompile (FileNameList, FileNames);

	if (NumOfFiles==0)
		return false;
		*/

	return true;
}

bool CL_MAGMA_RT::CreateKernel(const char *KernelName)
{
	if (!HasBeenInitialized)
	{
		printf ("Error: Uninitialized kernel\n");
		return false;
	}

	cl_program cpProgram = NULL;
	//printf ("getting kernel %s from %s\n", KernelName, Kernel2FileNamePool[string(KernelName)].c_str());
	cpProgram = ProgramPool[ Kernel2FileNamePool[string(KernelName)]];
	if (cpProgram==NULL)
	{
		printf ("Error: could not find program for kernel %s\n", KernelName);
		return false;
	}
	
	KernelPool[string(KernelName)] = clCreateKernel(cpProgram, KernelName, &ciErrNum);
	
	if (ciErrNum != CL_SUCCESS)
	{
		printf ("Error: could not create kernel %s\n", KernelName);
		return false;
	}
	
	//BigKernelPool[platform_id][dev_type] = KernelPool[string(KernelName)];
	//BigKernelPool[platform_id].insert(std::make_pair(dev_type, KernelPool[string(KernelName)]));
	//BigKernelPool[platform_id].insert(std::pair<cl_device_type, std::pair<string, cl_kernel> >(dev_type, KernelPool[string(KernelName)]));

	return true;
}

