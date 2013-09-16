#ifndef CL_MAGMA_RT_H
#define CL_MAGMA_RT_H
#pragma once

#include <vector>
#include <map>
#include <string>
#include <sys/stat.h>
#include <algorithm>

class cl_platform;

class CL_MAGMA_RT
{
	private:
		unsigned int MAX_GPU_COUNT;
		cl_uint ciDeviceCount;
		
		cl_platform_id cpPlatform;	//
		cl_device_id* cdDevices;        // OpenCL current device list    
		cl_context cxGPUContext;        // OpenCL current context (not only GPUs)
		cl_command_queue *commandQueue;
		//BELGIOVINE
		std::vector<cl_platform> cpPlatforms;
		cl_device_type dev_type;	// Current OpenCL devices type
		uint platform_id;		// Current OpenCL platform id
		//BELGIOVINE end
		
		cl_kernel ckKernel;             // OpenCL kernel
		cl_event ceEvent;               // OpenCL event
		
		size_t szParmDataBytes;         // Byte size of context information
		size_t szKernelLength;          // Byte size of kernel code
		cl_int ciErrNum;                // Error code var
		
		bool HasBeenInitialized;
		std::map<std::string, std::string> KernelMap;
		
		int GatherFilesToCompile(const char* FileNameList, std::vector<std::string>&);
		std::string fileToString(const char* FileName);
		
		//BELGIOVINE
		bool initDevices(const cl_platform_id src_platform, cl_device_id** devices, cl_context* context, cl_uint* num, cl_command_queue** queue, cl_device_type device_type, cl_uint max_ndev, cl_int* ciErrNum, char* label);
		cl_int initPlatform(const cl_platform_id src_platform);
		//BELGIOVINE end

		CL_MAGMA_RT();                                 // Private constructor
		~CL_MAGMA_RT();
		
	public:

		// singleton class to guarentee only 1 instance of runtime
		static CL_MAGMA_RT * Instance()
		{
			static CL_MAGMA_RT rrt;
			return &rrt;
		}
		cl_platform_id GetPlatform(); //return current platform in use
		
		//BELGIOVINE
		cl_platform_id SetPlatform(uint platformid, cl_device_type device_type); //SET platform (if platformid exists) and returns its cl_platform_id
		size_t GetNumPlatform(); //it returns the number of available platforms
		size_t GetNumDevices(); //it returns the number of currently used devices
		magma_queue_t * GetQueuePtr();
		bool generateKernelMap();
		uint GetPlatformIndex();
		//BELGIOVINE end
		
		cl_device_id * GetDevicePtr();
		cl_context GetContext();
		cl_command_queue GetCommandQueue(int queueid);
		bool Init ();
		//BELGIOVINE
		bool InitAll();
		//BELGIOVINE end
		bool Init(cl_platform_id gPlatform, cl_context gContext);
		bool Quit ();
		
		bool CompileFile(const char*FileName);
		bool CompileSourceFiles(const char* FileNameList);
		const char* GetErrorCode(cl_int err);
		bool BuildFromBinaries(const char*FileName);
		bool BuildKernelMap(const char* FileNameList);
		bool CreateKernel(const char* KernelName);
		
		std::map<std::string, std::string> Kernel2FileNamePool;	// kernel name -> file name 
		std::map<std::string, cl_program> ProgramPool;	// file name -> program
		//std::map<std::string, cl_kernel> KernelPool;	// kernel name -> kernel
		
		//BELGIOVI
		void PrintKernelMap();
		
		typedef std::map<std::string, cl_kernel> kernel_map;
		typedef std::map<cl_device_type, kernel_map> devType_map;
		typedef std::map<uint, devType_map> platform_map;
		
		kernel_map KernelPool;	// kernel name -> kernel
		platform_map BigKernelPool;
		
		cl_kernel getKernel(const std::string kernel_name);
				
		//BELGIOVI end
		//std::map<uint, std::map<cl_device_type, std::map<std::string, cl_kernel> > > BigKernelPool;
		//BigKernelPool [platform_id] [CL_DEVICE_TYPE_CPU/GPU] ["kernel_name"];
		
		//http://www.dreamincode.net/forums/topic/67804-c-multidimensional-associative-arrays/
		//http://www.yolinux.com/TUTORIALS/CppStlMultiMap.html

};

extern CL_MAGMA_RT *rt;

//BELGIOVINE
//#include <starpu.h>
//#include <starpu_opencl.h>

#include "magma.h"

//--- OPENCL PLATFORM CLASS --- //


class cl_platform {
	
	private:
		cl_platform_id 		platform;
		cl_uint 		n_gpu;
		cl_uint 		n_cpu;
		cl_uint 		n_acc;
		cl_context	 	gpu_context;
		cl_context	 	cpu_context;
		cl_context	 	acc_context;
		cl_device_id*		gpu_devices;	//GPU devices
		cl_command_queue*	gpu_queue;
		cl_device_id*		cpu_devices;	//CPU	
		cl_command_queue*	cpu_queue;
		cl_device_id*		acc_devices;	//ACCELERATOR
		cl_command_queue*	acc_queue;
	
	public:
		cl_platform();	//constructor
		cl_platform(const cl_platform &old_platform);	//copy constructor
		~cl_platform(); //destructor
		friend struct FindPlatformID;
		friend cl_platform_id CL_MAGMA_RT::SetPlatform(uint platformid, cl_device_type device_type);
		friend bool CL_MAGMA_RT::Init ();
		friend bool CL_MAGMA_RT::InitAll();
};

//classe per la ricerca dell'indice della piattaforma
struct FindPlatformID {
    const cl_platform_id platform;
    FindPlatformID(const cl_platform_id& ptr) : platform(ptr) {}
    bool operator()(const cl_platform& i) const { 
        return i.platform == platform; 
    }
};

bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}
//BELGIOVINE end

#endif        //  #ifndef CL_MAGMA_RT_H
