/* Generated by Cython 0.25.1 */

#ifndef __PYX_HAVE__Haar_cascade_detection
#define __PYX_HAVE__Haar_cascade_detection


#ifndef __PYX_HAVE_API__Haar_cascade_detection

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

__PYX_EXTERN_C DL_IMPORT(PyObject) *SetClassifiers(char const *, char const *);
__PYX_EXTERN_C DL_IMPORT(void) DetectFaceAndEyes(PyObject *, int);
__PYX_EXTERN_C DL_IMPORT(void) Run(char const *, char const *, int);

#endif /* !__PYX_HAVE_API__Haar_cascade_detection */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initHaar_cascade_detection(void);
#else
PyMODINIT_FUNC PyInit_Haar_cascade_detection(void);
#endif

#endif /* !__PYX_HAVE__Haar_cascade_detection */