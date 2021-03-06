#//////////////////////////////////////////////////////////////////////////////
#   -- clMAGMA (version 1.0.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      April 2012
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

INC += -I$(MAGMA_DIR)/control

ZSRC  = magmablas_z.cpp \
        zpermute.cpp    \
		zinplace_transpose.cpp \
		ztranspose.cpp   \
		ztranspose2.cpp  \
		zauxiliary.cpp  \
		zlacpy.cpp  \
		zswap.cpp		\
		\
        zinplace_transpose.cl \
        zpermute-v2.cl   \
		ztranspose.cl    \
        ztranspose-v2.cl \
        zauxiliary.cl \
		zlacpy.cl	  \
		zswap.cl	  \

-include Makefile.src
-include Makefile.local

ALLSRC = \
	interface.cpp \
	strerror.cpp \
	CL_MAGMA_RT.cpp  \
	$(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ = $(ALLSRC:.cpp=.o)

CL_SRCS = \
	sinplace_transpose.cl	\
	spermute-v2.cl		\
	stranspose.cl		\
	stranspose-v2.cl	\
	sauxiliary.cl	\
	slacpy.cl	  \
	sswap.cl	  \
	dinplace_transpose.cl	\
	dpermute-v2.cl		\
	dtranspose.cl		\
	dtranspose-v2.cl	\
	dauxiliary.cl	\
	dlacpy.cl	  \
	dswap.cl	  \
	cinplace_transpose.cl	\
	cpermute-v2.cl		\
	ctranspose.cl		\
	ctranspose-v2.cl	\
	cauxiliary.cl	\
	clacpy.cl	  \
	cswap.cl	  \
	zinplace_transpose.cl	\
	zpermute-v2.cl		\
	ztranspose.cl		\
	ztranspose-v2.cl	\
	zauxiliary.cl	\
	zlacpy.cl	  \
	zswap.cl	  \

CL_OBJS = $(CL_SRCS:.cl=.co)

CL_COMPILER = clcompile

all: $(LIBMAGMA) $(CL_OBJS) $(CL_COMPILER)

$(CL_COMPILER): clcompile.o #$(LIBMAGMA)
	$(CC) $(LDOPTS) $< -o $@ \
	-L../lib -lclmagma $(LIBDIR) $(LIB)

%.co: %.cl $(CL_COMPILER)
	./$(CL_COMPILER) $<

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

.f.o:
	$(FORT) $(OPTS) $(INC) -c $< -o $@

clean:
	rm -f *.o *.co *~ *.linkinfo $(CL_COMPILER)
	rm -rf cl_build/

cleanall: clean
	rm -f *.a core.*
	rm -f $(LIBMAGMA)
