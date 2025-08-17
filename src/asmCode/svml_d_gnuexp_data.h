#ifndef D_GNUEXP_DATA_H
#define D_GNUEXP_DATA_H

#define __gnudbT                           0
#define __dbInvLn2                      8192
#define __dbShifter                     8256
#define __dbLn2hi                       8320
#define __dbLn2lo                       8384
#define __dPC0                          8448
#define __dPC1                          8512
#define __dPC2                          8576
#define __lIndexMask                    8640
#define __iAbsMask                      8704
#define __iDomainRange                  8768

.macro gnu_double_vector offset value
.if .-__svml_dgnuexp_data != \offset
.err
.endif
.rept 8
.quad \value
.endr
.endm

.macro gnu_float_vector offset value
.if .-__svml_dgnuexp_data != \offset
.err
.endif
.rept 16
.long \value
.endr
.endm

#endif
