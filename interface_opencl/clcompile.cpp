#include <stdio.h>
#include <stdlib.h>

#include "CL_MAGMA_RT.h"

int main( int argc, char** argv )
{
	if ( argc != 2 ) {
		printf( "Usage: %s file.cl\n", argv[0] );
		exit(1);
	}
	
	CL_MAGMA_RT *runtime = CL_MAGMA_RT::Instance();
	runtime->Init();
	
	uint i;
	cl_platform_id arch_compiling = NULL;
	for(i = 0; i < runtime->GetNumPlatform(); i++){
	  printf("prima di SetPlatform ok\n" );
	  arch_compiling = runtime->SetPlatform(i, CL_DEVICE_TYPE_GPU);
	  printf("dopo di SetPlatform ok\n" );
	  if (arch_compiling != NULL) runtime->CompileFile( argv[1] );
	  arch_compiling = runtime->SetPlatform(i, CL_DEVICE_TYPE_CPU);
	  if (arch_compiling != NULL) runtime->CompileFile( argv[1] );
	  arch_compiling = NULL;
	}
	
	runtime->Quit();

	return 0;
}
