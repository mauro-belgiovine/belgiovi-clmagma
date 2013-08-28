/*
 *   -- clMAGMA (version 1.0.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      April 2012
 *
 * @author Mark Gates
 * @precisions normal z -> s d c
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "common_magma.h"

extern "C"
void magma_xerbla(const char *srname , magma_int_t info)
{
/*  -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012

    Purpose
    =======

    magma_xerbla is an error handler for the MAGMA routines.
    It is called by a MAGMA routine if an input parameter has an
    invalid value. It calls the LAPACK XERBLA routine, which by default
    prints an error message and stops execution.

    Installers may consider modifying the STOP statement in order to
    call system-specific exception-handling facilities.

    Arguments
    =========

    SRNAME  (input) CHARACTER*(*)
            The name of the routine which called XERBLA.
            In C it is convenient to use __func__.

    INFO    (input) INTEGER
            The position of the invalid parameter in the parameter list
            of the calling routine.

    =====================================================================   */

    // This assumes Fortran calling convention for strings as passing
    // length to string at end of argument list.
    // Different compilers have different conventions.
    int len = strlen( srname );
    lapackf77_xerbla( srname, &info, len );
}
