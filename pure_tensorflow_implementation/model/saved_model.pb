é­
ō%×%
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
ī
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
/
Atan
x"T
y"T"
Ttype:

2	
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
ģ
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
1
Square
x"T
y"T"
Ttype:

2	
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.9.02
b'unknown'¤Ļ
o
truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
¢
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*&
_output_shapes
:*
seed2 *

seed *
T0*
dtype0

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:

Variable
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
¬
Variable/AssignAssignVariabletruncated_normal*&
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
q
Variable/readIdentityVariable*&
_output_shapes
:*
T0*
_class
loc:@Variable
q
truncated_normal_1/shapeConst*%
valueB"         $   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_1/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
¦
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
dtype0*&
_output_shapes
:$*
seed2 *

seed 

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
:$*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*&
_output_shapes
:$


Variable_1
VariableV2*
dtype0*&
_output_shapes
:$*
	container *
shape:$*
shared_name 
“
Variable_1/AssignAssign
Variable_1truncated_normal_1*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:$*
use_locking(*
T0
w
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*&
_output_shapes
:$
q
truncated_normal_2/shapeConst*%
valueB"      $   0   *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_2/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
¦
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*&
_output_shapes
:$0*
seed2 *

seed 

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*&
_output_shapes
:$0
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*&
_output_shapes
:$0*
T0


Variable_2
VariableV2*
shape:$0*
shared_name *
dtype0*&
_output_shapes
:$0*
	container 
“
Variable_2/AssignAssign
Variable_2truncated_normal_2*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:$0*
use_locking(*
T0
w
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*&
_output_shapes
:$0
q
truncated_normal_3/shapeConst*%
valueB"      0   @   *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
¦
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*&
_output_shapes
:0@*
seed2 *

seed *
T0*
dtype0

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*&
_output_shapes
:0@*
T0
{
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*&
_output_shapes
:0@*
T0


Variable_3
VariableV2*&
_output_shapes
:0@*
	container *
shape:0@*
shared_name *
dtype0
“
Variable_3/AssignAssign
Variable_3truncated_normal_3*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:0@
w
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*&
_output_shapes
:0@
q
truncated_normal_4/shapeConst*
_output_shapes
:*%
valueB"      @   @   *
dtype0
\
truncated_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_4/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
¦
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*&
_output_shapes
:@@*
seed2 *

seed *
T0

truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*&
_output_shapes
:@@
{
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*&
_output_shapes
:@@


Variable_4
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
“
Variable_4/AssignAssign
Variable_4truncated_normal_4*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:@@
w
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*&
_output_shapes
:@@
i
truncated_normal_5/shapeConst*
valueB"  d   *
dtype0*
_output_shapes
:
\
truncated_normal_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_5/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 

"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
T0*
dtype0*
_output_shapes
:		d*
seed2 *

seed 

truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0*
_output_shapes
:		d
t
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0*
_output_shapes
:		d


Variable_5
VariableV2*
dtype0*
_output_shapes
:		d*
	container *
shape:		d*
shared_name 
­
Variable_5/AssignAssign
Variable_5truncated_normal_5*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:		d*
use_locking(*
T0
p
Variable_5/readIdentity
Variable_5*
_output_shapes
:		d*
T0*
_class
loc:@Variable_5
i
truncated_normal_6/shapeConst*
valueB"d   2   *
dtype0*
_output_shapes
:
\
truncated_normal_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_6/stddevConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 

"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*
T0*
dtype0*
_output_shapes

:d2*
seed2 *

seed 

truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0*
_output_shapes

:d2
s
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
_output_shapes

:d2*
T0
~

Variable_6
VariableV2*
shape
:d2*
shared_name *
dtype0*
_output_shapes

:d2*
	container 
¬
Variable_6/AssignAssign
Variable_6truncated_normal_6*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:d2
o
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes

:d2
i
truncated_normal_7/shapeConst*
valueB"2   
   *
dtype0*
_output_shapes
:
\
truncated_normal_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_7/stddevConst*
_output_shapes
: *
valueB
 *ĶĢĢ=*
dtype0

"truncated_normal_7/TruncatedNormalTruncatedNormaltruncated_normal_7/shape*
_output_shapes

:2
*
seed2 *

seed *
T0*
dtype0

truncated_normal_7/mulMul"truncated_normal_7/TruncatedNormaltruncated_normal_7/stddev*
T0*
_output_shapes

:2

s
truncated_normal_7Addtruncated_normal_7/multruncated_normal_7/mean*
_output_shapes

:2
*
T0
~

Variable_7
VariableV2*
dtype0*
_output_shapes

:2
*
	container *
shape
:2
*
shared_name 
¬
Variable_7/AssignAssign
Variable_7truncated_normal_7*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes

:2

o
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes

:2

i
truncated_normal_8/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
\
truncated_normal_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_8/stddevConst*
_output_shapes
: *
valueB
 *ĶĢĢ=*
dtype0

"truncated_normal_8/TruncatedNormalTruncatedNormaltruncated_normal_8/shape*
_output_shapes

:
*
seed2 *

seed *
T0*
dtype0

truncated_normal_8/mulMul"truncated_normal_8/TruncatedNormaltruncated_normal_8/stddev*
_output_shapes

:
*
T0
s
truncated_normal_8Addtruncated_normal_8/multruncated_normal_8/mean*
T0*
_output_shapes

:

~

Variable_8
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
¬
Variable_8/AssignAssign
Variable_8truncated_normal_8*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes

:

o
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8*
_output_shapes

:

]
random_normal/shapeConst*
valueB:*
dtype0*
_output_shapes
:
W
random_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
Y
random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
T0*
dtype0*
_output_shapes
:*
seed2 *

seed 
w
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes
:
`
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes
:
v

Variable_9
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
£
Variable_9/AssignAssign
Variable_9random_normal*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes
:
k
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9*
_output_shapes
:
_
random_normal_1/shapeConst*
valueB:$*
dtype0*
_output_shapes
:
Y
random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_1/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*

seed *
T0*
dtype0*
_output_shapes
:$*
seed2 
}
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes
:$
f
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*
_output_shapes
:$
w
Variable_10
VariableV2*
shared_name *
dtype0*
_output_shapes
:$*
	container *
shape:$
Ø
Variable_10/AssignAssignVariable_10random_normal_1*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:$
n
Variable_10/readIdentityVariable_10*
_output_shapes
:$*
T0*
_class
loc:@Variable_10
_
random_normal_2/shapeConst*
dtype0*
_output_shapes
:*
valueB:0
Y
random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_2/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
dtype0*
_output_shapes
:0*
seed2 *

seed *
T0
}
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
_output_shapes
:0*
T0
f
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0*
_output_shapes
:0
w
Variable_11
VariableV2*
dtype0*
_output_shapes
:0*
	container *
shape:0*
shared_name 
Ø
Variable_11/AssignAssignVariable_11random_normal_2*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes
:0
n
Variable_11/readIdentityVariable_11*
_output_shapes
:0*
T0*
_class
loc:@Variable_11
_
random_normal_3/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
Y
random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_3/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
_output_shapes
:@*
seed2 *

seed *
T0*
dtype0
}
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
_output_shapes
:@*
T0
f
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
_output_shapes
:@*
T0
w
Variable_12
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
Ø
Variable_12/AssignAssignVariable_12random_normal_3*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*
_output_shapes
:@
n
Variable_12/readIdentityVariable_12*
T0*
_class
loc:@Variable_12*
_output_shapes
:@
_
random_normal_4/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
Y
random_normal_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_4/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*

seed *
T0*
dtype0*
_output_shapes
:@*
seed2 
}
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
_output_shapes
:@*
T0
f
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0*
_output_shapes
:@
w
Variable_13
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
Ø
Variable_13/AssignAssignVariable_13random_normal_4*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes
:@
n
Variable_13/readIdentityVariable_13*
_output_shapes
:@*
T0*
_class
loc:@Variable_13
_
random_normal_5/shapeConst*
valueB:d*
dtype0*
_output_shapes
:
Y
random_normal_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_5/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
T0*
dtype0*
_output_shapes
:d*
seed2 *

seed 
}
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
_output_shapes
:d*
T0
f
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
_output_shapes
:d*
T0
w
Variable_14
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
Ø
Variable_14/AssignAssignVariable_14random_normal_5*
_output_shapes
:d*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(
n
Variable_14/readIdentityVariable_14*
T0*
_class
loc:@Variable_14*
_output_shapes
:d
_
random_normal_6/shapeConst*
valueB:2*
dtype0*
_output_shapes
:
Y
random_normal_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_6/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
T0*
dtype0*
_output_shapes
:2*
seed2 *

seed 
}
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0*
_output_shapes
:2
f
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0*
_output_shapes
:2
w
Variable_15
VariableV2*
shared_name *
dtype0*
_output_shapes
:2*
	container *
shape:2
Ø
Variable_15/AssignAssignVariable_15random_normal_6*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0*
_class
loc:@Variable_15
n
Variable_15/readIdentityVariable_15*
T0*
_class
loc:@Variable_15*
_output_shapes
:2
_
random_normal_7/shapeConst*
valueB:
*
dtype0*
_output_shapes
:
Y
random_normal_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_7/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*

seed *
T0*
dtype0*
_output_shapes
:
*
seed2 
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
_output_shapes
:
*
T0
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
_output_shapes
:
*
T0
w
Variable_16
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
Ø
Variable_16/AssignAssignVariable_16random_normal_7*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:
*
use_locking(
n
Variable_16/readIdentityVariable_16*
_output_shapes
:
*
T0*
_class
loc:@Variable_16
_
random_normal_8/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_8/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_8/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
T0*
dtype0*
_output_shapes
:*
seed2 *

seed 
}
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
T0*
_output_shapes
:
f
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
_output_shapes
:*
T0
w
Variable_17
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ø
Variable_17/AssignAssignVariable_17random_normal_8*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
n
Variable_17/readIdentityVariable_17*
_output_shapes
:*
T0*
_class
loc:@Variable_17
|
myInputPlaceholder*
dtype0*0
_output_shapes
:’’’’’’’’’BČ*%
shape:’’’’’’’’’BČ
n
PlaceholderPlaceholder*
shape:’’’’’’’’’*
dtype0*'
_output_shapes
:’’’’’’’’’
R
Placeholder_1Placeholder*
dtype0*
_output_shapes
:*
shape:
Ń
Conv2DConv2DmyInputVariable/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:’’’’’’’’’b*
	dilations

|
BiasAddBiasAddConv2DVariable_9/read*
T0*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’b
O
ReluReluBiasAdd*/
_output_shapes
:’’’’’’’’’b*
T0
Ņ
Conv2D_1Conv2DReluVariable_1/read*/
_output_shapes
:’’’’’’’’’/$*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

	BiasAdd_1BiasAddConv2D_1Variable_10/read*/
_output_shapes
:’’’’’’’’’/$*
T0*
data_formatNHWC
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:’’’’’’’’’/$
Ō
Conv2D_2Conv2DRelu_1Variable_2/read*
paddingVALID*/
_output_shapes
:’’’’’’’’’0*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

	BiasAdd_2BiasAddConv2D_2Variable_11/read*
T0*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’0
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:’’’’’’’’’0
Ō
Conv2D_3Conv2DRelu_2Variable_3/read*
paddingVALID*/
_output_shapes
:’’’’’’’’’@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

	BiasAdd_3BiasAddConv2D_3Variable_12/read*
T0*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’@
S
Relu_3Relu	BiasAdd_3*/
_output_shapes
:’’’’’’’’’@*
T0
Ō
Conv2D_4Conv2DRelu_3Variable_4/read*/
_output_shapes
:’’’’’’’’’@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

	BiasAdd_4BiasAddConv2D_4Variable_13/read*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’@*
T0
S
Relu_4Relu	BiasAdd_4*
T0*/
_output_shapes
:’’’’’’’’’@
[
Flatten/flatten/ShapeShapeRelu_4*
T0*
out_type0*
_output_shapes
:
m
#Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%Flatten/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
É
Flatten/flatten/strided_sliceStridedSliceFlatten/flatten/Shape#Flatten/flatten/strided_slice/stack%Flatten/flatten/strided_slice/stack_1%Flatten/flatten/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
j
Flatten/flatten/Reshape/shape/1Const*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

Flatten/flatten/Reshape/shapePackFlatten/flatten/strided_sliceFlatten/flatten/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0

Flatten/flatten/ReshapeReshapeRelu_4Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:’’’’’’’’’	

MatMulMatMulFlatten/flatten/ReshapeVariable_5/read*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b( *
T0
V
AddAddMatMulVariable_14/read*
T0*'
_output_shapes
:’’’’’’’’’d
E
Relu_5ReluAdd*
T0*'
_output_shapes
:’’’’’’’’’d

MatMul_1MatMulRelu_5Variable_6/read*
T0*'
_output_shapes
:’’’’’’’’’2*
transpose_a( *
transpose_b( 
Z
Add_1AddMatMul_1Variable_15/read*
T0*'
_output_shapes
:’’’’’’’’’2
G
Relu_6ReluAdd_1*
T0*'
_output_shapes
:’’’’’’’’’2

MatMul_2MatMulRelu_6Variable_7/read*
transpose_b( *
T0*'
_output_shapes
:’’’’’’’’’
*
transpose_a( 
Z
Add_2AddMatMul_2Variable_16/read*
T0*'
_output_shapes
:’’’’’’’’’

G
Relu_7ReluAdd_2*
T0*'
_output_shapes
:’’’’’’’’’


MatMul_3MatMulRelu_7Variable_8/read*
transpose_b( *
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( 
Z
Add_3AddMatMul_3Variable_17/read*'
_output_shapes
:’’’’’’’’’*
T0
E
AtanAtanAdd_3*
T0*'
_output_shapes
:’’’’’’’’’
J
Mul/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
I
MulMulAtanMul/y*
T0*'
_output_shapes
:’’’’’’’’’
K
myOutputIdentityMul*
T0*'
_output_shapes
:’’’’’’’’’
N
SubSubPlaceholderMul*'
_output_shapes
:’’’’’’’’’*
T0
G
SquareSquareSub*'
_output_shapes
:’’’’’’’’’*
T0
V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
Y
MeanMeanSquareConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
@
L2LossL2LossVariable/read*
T0*
_output_shapes
: 
D
L2Loss_1L2LossVariable_1/read*
T0*
_output_shapes
: 
D
L2Loss_2L2LossVariable_2/read*
T0*
_output_shapes
: 
D
L2Loss_3L2LossVariable_3/read*
T0*
_output_shapes
: 
D
L2Loss_4L2LossVariable_4/read*
_output_shapes
: *
T0
D
L2Loss_5L2LossVariable_5/read*
T0*
_output_shapes
: 
D
L2Loss_6L2LossVariable_6/read*
_output_shapes
: *
T0
D
L2Loss_7L2LossVariable_7/read*
_output_shapes
: *
T0
D
L2Loss_8L2LossVariable_8/read*
T0*
_output_shapes
: 
D
L2Loss_9L2LossVariable_9/read*
_output_shapes
: *
T0
F
	L2Loss_10L2LossVariable_10/read*
T0*
_output_shapes
: 
F
	L2Loss_11L2LossVariable_11/read*
_output_shapes
: *
T0
F
	L2Loss_12L2LossVariable_12/read*
T0*
_output_shapes
: 
F
	L2Loss_13L2LossVariable_13/read*
_output_shapes
: *
T0
F
	L2Loss_14L2LossVariable_14/read*
T0*
_output_shapes
: 
F
	L2Loss_15L2LossVariable_15/read*
T0*
_output_shapes
: 
F
	L2Loss_16L2LossVariable_16/read*
_output_shapes
: *
T0
F
	L2Loss_17L2LossVariable_17/read*
T0*
_output_shapes
: 
š
AddNAddNL2LossL2Loss_1L2Loss_2L2Loss_3L2Loss_4L2Loss_5L2Loss_6L2Loss_7L2Loss_8L2Loss_9	L2Loss_10	L2Loss_11	L2Loss_12	L2Loss_13	L2Loss_14	L2Loss_15	L2Loss_16	L2Loss_17*
T0*
N*
_output_shapes
: 
L
mul_1/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
<
mul_1MulAddNmul_1/y*
T0*
_output_shapes
: 
:
add_4AddMeanmul_1*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
>
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/Fill
µ
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/add_4_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
·
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/Fill&^gradients/add_4_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Æ
gradients/Mean_grad/ReshapeReshape-gradients/add_4_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
_
gradients/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:’’’’’’’’’*

Tmultiples0*
T0
a
gradients/Mean_grad/Shape_1ShapeSquare*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:’’’’’’’’’*
T0
z
gradients/mul_1_grad/MulMul/gradients/add_4_grad/tuple/control_dependency_1mul_1/y*
T0*
_output_shapes
: 
y
gradients/mul_1_grad/Mul_1Mul/gradients/add_4_grad/tuple/control_dependency_1AddN*
T0*
_output_shapes
: 
e
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Mul^gradients/mul_1_grad/Mul_1
É
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Mul&^gradients/mul_1_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
Ļ
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Mul_1&^gradients/mul_1_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_1_grad/Mul_1*
_output_shapes
: *
T0
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
t
gradients/Square_grad/MulMulSubgradients/Square_grad/Const*
T0*'
_output_shapes
:’’’’’’’’’

gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0*'
_output_shapes
:’’’’’’’’’
\
$gradients/AddN_grad/tuple/group_depsNoOp.^gradients/mul_1_grad/tuple/control_dependency
Ü
,gradients/AddN_grad/tuple/control_dependencyIdentity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
Ž
.gradients/AddN_grad/tuple/control_dependency_1Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
Ž
.gradients/AddN_grad/tuple/control_dependency_2Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
Ž
.gradients/AddN_grad/tuple/control_dependency_3Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_1_grad/Mul
Ž
.gradients/AddN_grad/tuple/control_dependency_4Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
Ž
.gradients/AddN_grad/tuple/control_dependency_5Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
Ž
.gradients/AddN_grad/tuple/control_dependency_6Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_1_grad/Mul
Ž
.gradients/AddN_grad/tuple/control_dependency_7Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
Ž
.gradients/AddN_grad/tuple/control_dependency_8Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
Ž
.gradients/AddN_grad/tuple/control_dependency_9Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_1_grad/Mul
ß
/gradients/AddN_grad/tuple/control_dependency_10Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
ß
/gradients/AddN_grad/tuple/control_dependency_11Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
ß
/gradients/AddN_grad/tuple/control_dependency_12Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_1_grad/Mul
ß
/gradients/AddN_grad/tuple/control_dependency_13Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
ß
/gradients/AddN_grad/tuple/control_dependency_14Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
ß
/gradients/AddN_grad/tuple/control_dependency_15Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_1_grad/Mul
ß
/gradients/AddN_grad/tuple/control_dependency_16Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
ß
/gradients/AddN_grad/tuple/control_dependency_17Identity-gradients/mul_1_grad/tuple/control_dependency%^gradients/AddN_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_1_grad/Mul
c
gradients/Sub_grad/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
]
gradients/Sub_grad/Shape_1ShapeMul*
T0*
out_type0*
_output_shapes
:
“
(gradients/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Sub_grad/Shapegradients/Sub_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¤
gradients/Sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/Sub_grad/ReshapeReshapegradients/Sub_grad/Sumgradients/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
Ø
gradients/Sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/Sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/Sub_grad/NegNeggradients/Sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/Sub_grad/Reshape_1Reshapegradients/Sub_grad/Neggradients/Sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
g
#gradients/Sub_grad/tuple/group_depsNoOp^gradients/Sub_grad/Reshape^gradients/Sub_grad/Reshape_1
Ś
+gradients/Sub_grad/tuple/control_dependencyIdentitygradients/Sub_grad/Reshape$^gradients/Sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Sub_grad/Reshape*'
_output_shapes
:’’’’’’’’’
ą
-gradients/Sub_grad/tuple/control_dependency_1Identitygradients/Sub_grad/Reshape_1$^gradients/Sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Sub_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’*
T0

gradients/L2Loss_grad/mulMulVariable/read,gradients/AddN_grad/tuple/control_dependency*&
_output_shapes
:*
T0

gradients/L2Loss_1_grad/mulMulVariable_1/read.gradients/AddN_grad/tuple/control_dependency_1*&
_output_shapes
:$*
T0

gradients/L2Loss_2_grad/mulMulVariable_2/read.gradients/AddN_grad/tuple/control_dependency_2*&
_output_shapes
:$0*
T0

gradients/L2Loss_3_grad/mulMulVariable_3/read.gradients/AddN_grad/tuple/control_dependency_3*&
_output_shapes
:0@*
T0

gradients/L2Loss_4_grad/mulMulVariable_4/read.gradients/AddN_grad/tuple/control_dependency_4*
T0*&
_output_shapes
:@@

gradients/L2Loss_5_grad/mulMulVariable_5/read.gradients/AddN_grad/tuple/control_dependency_5*
T0*
_output_shapes
:		d

gradients/L2Loss_6_grad/mulMulVariable_6/read.gradients/AddN_grad/tuple/control_dependency_6*
_output_shapes

:d2*
T0

gradients/L2Loss_7_grad/mulMulVariable_7/read.gradients/AddN_grad/tuple/control_dependency_7*
T0*
_output_shapes

:2


gradients/L2Loss_8_grad/mulMulVariable_8/read.gradients/AddN_grad/tuple/control_dependency_8*
T0*
_output_shapes

:


gradients/L2Loss_9_grad/mulMulVariable_9/read.gradients/AddN_grad/tuple/control_dependency_9*
_output_shapes
:*
T0

gradients/L2Loss_10_grad/mulMulVariable_10/read/gradients/AddN_grad/tuple/control_dependency_10*
T0*
_output_shapes
:$

gradients/L2Loss_11_grad/mulMulVariable_11/read/gradients/AddN_grad/tuple/control_dependency_11*
_output_shapes
:0*
T0

gradients/L2Loss_12_grad/mulMulVariable_12/read/gradients/AddN_grad/tuple/control_dependency_12*
T0*
_output_shapes
:@

gradients/L2Loss_13_grad/mulMulVariable_13/read/gradients/AddN_grad/tuple/control_dependency_13*
_output_shapes
:@*
T0

gradients/L2Loss_14_grad/mulMulVariable_14/read/gradients/AddN_grad/tuple/control_dependency_14*
_output_shapes
:d*
T0

gradients/L2Loss_15_grad/mulMulVariable_15/read/gradients/AddN_grad/tuple/control_dependency_15*
T0*
_output_shapes
:2

gradients/L2Loss_16_grad/mulMulVariable_16/read/gradients/AddN_grad/tuple/control_dependency_16*
T0*
_output_shapes
:


gradients/L2Loss_17_grad/mulMulVariable_17/read/gradients/AddN_grad/tuple/control_dependency_17*
_output_shapes
:*
T0
\
gradients/Mul_grad/ShapeShapeAtan*
_output_shapes
:*
T0*
out_type0
]
gradients/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
“
(gradients/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_grad/Shapegradients/Mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/Mul_grad/MulMul-gradients/Sub_grad/tuple/control_dependency_1Mul/y*
T0*'
_output_shapes
:’’’’’’’’’

gradients/Mul_grad/SumSumgradients/Mul_grad/Mul(gradients/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/Mul_grad/ReshapeReshapegradients/Mul_grad/Sumgradients/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

gradients/Mul_grad/Mul_1MulAtan-gradients/Sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:’’’’’’’’’
„
gradients/Mul_grad/Sum_1Sumgradients/Mul_grad/Mul_1*gradients/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/Mul_grad/Reshape_1Reshapegradients/Mul_grad/Sum_1gradients/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/Mul_grad/tuple/group_depsNoOp^gradients/Mul_grad/Reshape^gradients/Mul_grad/Reshape_1
Ś
+gradients/Mul_grad/tuple/control_dependencyIdentitygradients/Mul_grad/Reshape$^gradients/Mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Mul_grad/Reshape*'
_output_shapes
:’’’’’’’’’
Ļ
-gradients/Mul_grad/tuple/control_dependency_1Identitygradients/Mul_grad/Reshape_1$^gradients/Mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Mul_grad/Reshape_1*
_output_shapes
: 

gradients/Atan_grad/SquareSquareAdd_3,^gradients/Mul_grad/tuple/control_dependency*'
_output_shapes
:’’’’’’’’’*
T0

gradients/Atan_grad/ConstConst,^gradients/Mul_grad/tuple/control_dependency*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients/Atan_grad/AddAddgradients/Atan_grad/Constgradients/Atan_grad/Square*
T0*'
_output_shapes
:’’’’’’’’’
w
gradients/Atan_grad/Reciprocal
Reciprocalgradients/Atan_grad/Add*
T0*'
_output_shapes
:’’’’’’’’’

gradients/Atan_grad/mulMul+gradients/Mul_grad/tuple/control_dependencygradients/Atan_grad/Reciprocal*'
_output_shapes
:’’’’’’’’’*
T0
b
gradients/Add_3_grad/ShapeShapeMatMul_3*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ŗ
*gradients/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_3_grad/Shapegradients/Add_3_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¤
gradients/Add_3_grad/SumSumgradients/Atan_grad/mul*gradients/Add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/Add_3_grad/ReshapeReshapegradients/Add_3_grad/Sumgradients/Add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
Ø
gradients/Add_3_grad/Sum_1Sumgradients/Atan_grad/mul,gradients/Add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/Add_3_grad/Reshape_1Reshapegradients/Add_3_grad/Sum_1gradients/Add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/Add_3_grad/tuple/group_depsNoOp^gradients/Add_3_grad/Reshape^gradients/Add_3_grad/Reshape_1
ā
-gradients/Add_3_grad/tuple/control_dependencyIdentitygradients/Add_3_grad/Reshape&^gradients/Add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_3_grad/Reshape*'
_output_shapes
:’’’’’’’’’
Ū
/gradients/Add_3_grad/tuple/control_dependency_1Identitygradients/Add_3_grad/Reshape_1&^gradients/Add_3_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/Add_3_grad/Reshape_1
Ą
gradients/MatMul_3_grad/MatMulMatMul-gradients/Add_3_grad/tuple/control_dependencyVariable_8/read*
T0*'
_output_shapes
:’’’’’’’’’
*
transpose_a( *
transpose_b(
°
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_7-gradients/Add_3_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
ģ
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*'
_output_shapes
:’’’’’’’’’

é
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
_output_shapes

:
*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1
Ä
gradients/AddNAddNgradients/L2Loss_17_grad/mul/gradients/Add_3_grad/tuple/control_dependency_1*
T0*/
_class%
#!loc:@gradients/L2Loss_17_grad/mul*
N*
_output_shapes
:

gradients/Relu_7_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_7*
T0*'
_output_shapes
:’’’’’’’’’

Ė
gradients/AddN_1AddNgradients/L2Loss_8_grad/mul2gradients/MatMul_3_grad/tuple/control_dependency_1*.
_class$
" loc:@gradients/L2Loss_8_grad/mul*
N*
_output_shapes

:
*
T0
b
gradients/Add_2_grad/ShapeShapeMatMul_2*
out_type0*
_output_shapes
:*
T0
f
gradients/Add_2_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
ŗ
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
«
gradients/Add_2_grad/SumSumgradients/Relu_7_grad/ReluGrad*gradients/Add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

Æ
gradients/Add_2_grad/Sum_1Sumgradients/Relu_7_grad/ReluGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
ā
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’
*
T0*/
_class%
#!loc:@gradients/Add_2_grad/Reshape
Ū
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1*
_output_shapes
:

Ą
gradients/MatMul_2_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_7/read*
T0*'
_output_shapes
:’’’’’’’’’2*
transpose_a( *
transpose_b(
°
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_6-gradients/Add_2_grad/tuple/control_dependency*
_output_shapes

:2
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
ģ
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’2*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul
é
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
_output_shapes

:2

Ę
gradients/AddN_2AddNgradients/L2Loss_16_grad/mul/gradients/Add_2_grad/tuple/control_dependency_1*
T0*/
_class%
#!loc:@gradients/L2Loss_16_grad/mul*
N*
_output_shapes
:


gradients/Relu_6_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_6*
T0*'
_output_shapes
:’’’’’’’’’2
Ė
gradients/AddN_3AddNgradients/L2Loss_7_grad/mul2gradients/MatMul_2_grad/tuple/control_dependency_1*.
_class$
" loc:@gradients/L2Loss_7_grad/mul*
N*
_output_shapes

:2
*
T0
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
_output_shapes
:*
T0*
out_type0
f
gradients/Add_1_grad/Shape_1Const*
valueB:2*
dtype0*
_output_shapes
:
ŗ
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
«
gradients/Add_1_grad/SumSumgradients/Relu_6_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’2
Æ
gradients/Add_1_grad/Sum_1Sumgradients/Relu_6_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:2
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
ā
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*'
_output_shapes
:’’’’’’’’’2*
T0
Ū
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
_output_shapes
:2
Ą
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependencyVariable_6/read*
T0*'
_output_shapes
:’’’’’’’’’d*
transpose_a( *
transpose_b(
°
 gradients/MatMul_1_grad/MatMul_1MatMulRelu_5-gradients/Add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:d2*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
ģ
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’d*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
é
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes

:d2*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
Ę
gradients/AddN_4AddNgradients/L2Loss_15_grad/mul/gradients/Add_1_grad/tuple/control_dependency_1*/
_class%
#!loc:@gradients/L2Loss_15_grad/mul*
N*
_output_shapes
:2*
T0

gradients/Relu_5_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu_5*'
_output_shapes
:’’’’’’’’’d*
T0
Ė
gradients/AddN_5AddNgradients/L2Loss_6_grad/mul2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@gradients/L2Loss_6_grad/mul*
N*
_output_shapes

:d2
^
gradients/Add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
gradients/Add_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:
“
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
§
gradients/Add_grad/SumSumgradients/Relu_5_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*'
_output_shapes
:’’’’’’’’’d*
T0*
Tshape0
«
gradients/Add_grad/Sum_1Sumgradients/Relu_5_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
Ś
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Add_grad/Reshape*'
_output_shapes
:’’’’’’’’’d
Ó
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
_output_shapes
:d
½
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencyVariable_5/read*
T0*(
_output_shapes
:’’’’’’’’’	*
transpose_a( *
transpose_b(
¾
gradients/MatMul_grad/MatMul_1MatMulFlatten/flatten/Reshape+gradients/Add_grad/tuple/control_dependency*
_output_shapes
:		d*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
å
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:’’’’’’’’’	*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
ā
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
:		d
Ä
gradients/AddN_6AddNgradients/L2Loss_14_grad/mul-gradients/Add_grad/tuple/control_dependency_1*
T0*/
_class%
#!loc:@gradients/L2Loss_14_grad/mul*
N*
_output_shapes
:d
r
,gradients/Flatten/flatten/Reshape_grad/ShapeShapeRelu_4*
_output_shapes
:*
T0*
out_type0
ß
.gradients/Flatten/flatten/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependency,gradients/Flatten/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:’’’’’’’’’@*
T0
Ź
gradients/AddN_7AddNgradients/L2Loss_5_grad/mul0gradients/MatMul_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@gradients/L2Loss_5_grad/mul*
N*
_output_shapes
:		d

gradients/Relu_4_grad/ReluGradReluGrad.gradients/Flatten/flatten/Reshape_grad/ReshapeRelu_4*
T0*/
_output_shapes
:’’’’’’’’’@

$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGradgradients/Relu_4_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
y
)gradients/BiasAdd_4_grad/tuple/group_depsNoOp%^gradients/BiasAdd_4_grad/BiasAddGrad^gradients/Relu_4_grad/ReluGrad
ö
1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentitygradients/Relu_4_grad/ReluGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*/
_output_shapes
:’’’’’’’’’@
ļ
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad*
_output_shapes
:@

gradients/Conv2D_4_grad/ShapeNShapeNRelu_3Variable_4/read*
T0*
out_type0*
N* 
_output_shapes
::
ź
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_4/read1gradients/BiasAdd_4_grad/tuple/control_dependency*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
å
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3 gradients/Conv2D_4_grad/ShapeN:11gradients/BiasAdd_4_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’

(gradients/Conv2D_4_grad/tuple/group_depsNoOp-^gradients/Conv2D_4_grad/Conv2DBackpropFilter,^gradients/Conv2D_4_grad/Conv2DBackpropInput

0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*/
_output_shapes
:’’’’’’’’’@

2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*&
_output_shapes
:@@*
T0*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter
Ź
gradients/AddN_8AddNgradients/L2Loss_13_grad/mul3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
T0*/
_class%
#!loc:@gradients/L2Loss_13_grad/mul*
N*
_output_shapes
:@

gradients/Relu_3_grad/ReluGradReluGrad0gradients/Conv2D_4_grad/tuple/control_dependencyRelu_3*
T0*/
_output_shapes
:’’’’’’’’’@
Ó
gradients/AddN_9AddNgradients/L2Loss_4_grad/mul2gradients/Conv2D_4_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@gradients/L2Loss_4_grad/mul*
N*&
_output_shapes
:@@

$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp%^gradients/BiasAdd_3_grad/BiasAddGrad^gradients/Relu_3_grad/ReluGrad
ö
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*/
_output_shapes
:’’’’’’’’’@*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
ļ
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
_output_shapes
:@*
T0*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad

gradients/Conv2D_3_grad/ShapeNShapeNRelu_2Variable_3/read*
N* 
_output_shapes
::*
T0*
out_type0
ź
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_3/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
	dilations

å
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2 gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’

(gradients/Conv2D_3_grad/tuple/group_depsNoOp-^gradients/Conv2D_3_grad/Conv2DBackpropFilter,^gradients/Conv2D_3_grad/Conv2DBackpropInput

0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*/
_output_shapes
:’’’’’’’’’0

2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*&
_output_shapes
:0@
Ė
gradients/AddN_10AddNgradients/L2Loss_12_grad/mul3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
T0*/
_class%
#!loc:@gradients/L2Loss_12_grad/mul*
N*
_output_shapes
:@

gradients/Relu_2_grad/ReluGradReluGrad0gradients/Conv2D_3_grad/tuple/control_dependencyRelu_2*
T0*/
_output_shapes
:’’’’’’’’’0
Ō
gradients/AddN_11AddNgradients/L2Loss_3_grad/mul2gradients/Conv2D_3_grad/tuple/control_dependency_1*&
_output_shapes
:0@*
T0*.
_class$
" loc:@gradients/L2Loss_3_grad/mul*
N

$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:0
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp%^gradients/BiasAdd_2_grad/BiasAddGrad^gradients/Relu_2_grad/ReluGrad
ö
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:’’’’’’’’’0
ļ
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:0

gradients/Conv2D_2_grad/ShapeNShapeNRelu_1Variable_2/read*
T0*
out_type0*
N* 
_output_shapes
::
ź
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_2/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
	dilations
*
T0
å
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1 gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’

(gradients/Conv2D_2_grad/tuple/group_depsNoOp-^gradients/Conv2D_2_grad/Conv2DBackpropFilter,^gradients/Conv2D_2_grad/Conv2DBackpropInput

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*/
_output_shapes
:’’’’’’’’’/$

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*&
_output_shapes
:$0*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
Ė
gradients/AddN_12AddNgradients/L2Loss_11_grad/mul3gradients/BiasAdd_2_grad/tuple/control_dependency_1*/
_class%
#!loc:@gradients/L2Loss_11_grad/mul*
N*
_output_shapes
:0*
T0

gradients/Relu_1_grad/ReluGradReluGrad0gradients/Conv2D_2_grad/tuple/control_dependencyRelu_1*/
_output_shapes
:’’’’’’’’’/$*
T0
Ō
gradients/AddN_13AddNgradients/L2Loss_2_grad/mul2gradients/Conv2D_2_grad/tuple/control_dependency_1*.
_class$
" loc:@gradients/L2Loss_2_grad/mul*
N*&
_output_shapes
:$0*
T0

$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:$*
T0
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp%^gradients/BiasAdd_1_grad/BiasAddGrad^gradients/Relu_1_grad/ReluGrad
ö
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*/
_output_shapes
:’’’’’’’’’/$
ļ
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:$

gradients/Conv2D_1_grad/ShapeNShapeNReluVariable_1/read* 
_output_shapes
::*
T0*
out_type0*
N
ź
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_1/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
	dilations
*
T0
ć
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

(gradients/Conv2D_1_grad/tuple/group_depsNoOp-^gradients/Conv2D_1_grad/Conv2DBackpropFilter,^gradients/Conv2D_1_grad/Conv2DBackpropInput

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:’’’’’’’’’b*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*&
_output_shapes
:$
Ė
gradients/AddN_14AddNgradients/L2Loss_10_grad/mul3gradients/BiasAdd_1_grad/tuple/control_dependency_1*/
_class%
#!loc:@gradients/L2Loss_10_grad/mul*
N*
_output_shapes
:$*
T0

gradients/Relu_grad/ReluGradReluGrad0gradients/Conv2D_1_grad/tuple/control_dependencyRelu*/
_output_shapes
:’’’’’’’’’b*
T0
Ō
gradients/AddN_15AddNgradients/L2Loss_1_grad/mul2gradients/Conv2D_1_grad/tuple/control_dependency_1*.
_class$
" loc:@gradients/L2Loss_1_grad/mul*
N*&
_output_shapes
:$*
T0

"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp#^gradients/BiasAdd_grad/BiasAddGrad^gradients/Relu_grad/ReluGrad
ī
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:’’’’’’’’’b*
T0
ē
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

gradients/Conv2D_grad/ShapeNShapeNmyInputVariable/read*
T0*
out_type0*
N* 
_output_shapes
::
ā
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
	dilations

ą
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermyInputgradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
paddingVALID*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

&gradients/Conv2D_grad/tuple/group_depsNoOp+^gradients/Conv2D_grad/Conv2DBackpropFilter*^gradients/Conv2D_grad/Conv2DBackpropInput

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:’’’’’’’’’BČ*
T0

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Ē
gradients/AddN_16AddNgradients/L2Loss_9_grad/mul1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@gradients/L2Loss_9_grad/mul*
N*
_output_shapes
:
Ī
gradients/AddN_17AddNgradients/L2Loss_grad/mul0gradients/Conv2D_grad/tuple/control_dependency_1*,
_class"
 loc:@gradients/L2Loss_grad/mul*
N*&
_output_shapes
:*
T0
{
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@Variable*
valueB
 *fff?

beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container *
shape: 
«
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@Variable
{
beta2_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *w¾?*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container 
«
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
„
/Variable/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable*%
valueB"            *
dtype0*
_output_shapes
:

%Variable/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable*
valueB
 *    *
dtype0*
_output_shapes
: 
ß
Variable/Adam/Initializer/zerosFill/Variable/Adam/Initializer/zeros/shape_as_tensor%Variable/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable*

index_type0*&
_output_shapes
:
®
Variable/Adam
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *
_class
loc:@Variable*
	container 
Å
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:
{
Variable/Adam/readIdentityVariable/Adam*&
_output_shapes
:*
T0*
_class
loc:@Variable
§
1Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@Variable*%
valueB"            *
dtype0

'Variable/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@Variable*
valueB
 *    *
dtype0
å
!Variable/Adam_1/Initializer/zerosFill1Variable/Adam_1/Initializer/zeros/shape_as_tensor'Variable/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable*

index_type0*&
_output_shapes
:
°
Variable/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape:*
dtype0*&
_output_shapes
:
Ė
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*&
_output_shapes
:
©
1Variable_1/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_1*%
valueB"         $   *
dtype0*
_output_shapes
:

'Variable_1/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ē
!Variable_1/Adam/Initializer/zerosFill1Variable_1/Adam/Initializer/zeros/shape_as_tensor'Variable_1/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_1*

index_type0*&
_output_shapes
:$
²
Variable_1/Adam
VariableV2*
dtype0*&
_output_shapes
:$*
shared_name *
_class
loc:@Variable_1*
	container *
shape:$
Ķ
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:$

Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*&
_output_shapes
:$
«
3Variable_1/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_1*%
valueB"         $   *
dtype0*
_output_shapes
:

)Variable_1/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_1*
valueB
 *    *
dtype0*
_output_shapes
: 
ķ
#Variable_1/Adam_1/Initializer/zerosFill3Variable_1/Adam_1/Initializer/zeros/shape_as_tensor)Variable_1/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_1*

index_type0*&
_output_shapes
:$
“
Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape:$*
dtype0*&
_output_shapes
:$
Ó
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:$

Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*&
_output_shapes
:$
©
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_2*%
valueB"      $   0   *
dtype0*
_output_shapes
:

'Variable_2/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_2*
valueB
 *    *
dtype0*
_output_shapes
: 
ē
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_2*

index_type0*&
_output_shapes
:$0
²
Variable_2/Adam
VariableV2*
dtype0*&
_output_shapes
:$0*
shared_name *
_class
loc:@Variable_2*
	container *
shape:$0
Ķ
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:$0*
use_locking(*
T0*
_class
loc:@Variable_2

Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*&
_output_shapes
:$0
«
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_2*%
valueB"      $   0   *
dtype0*
_output_shapes
:

)Variable_2/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_2*
valueB
 *    *
dtype0*
_output_shapes
: 
ķ
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
_class
loc:@Variable_2*

index_type0*&
_output_shapes
:$0*
T0
“
Variable_2/Adam_1
VariableV2*
	container *
shape:$0*
dtype0*&
_output_shapes
:$0*
shared_name *
_class
loc:@Variable_2
Ó
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:$0*
use_locking(*
T0

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*&
_output_shapes
:$0
©
1Variable_3/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_3*%
valueB"      0   @   *
dtype0*
_output_shapes
:

'Variable_3/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_3*
valueB
 *    *
dtype0*
_output_shapes
: 
ē
!Variable_3/Adam/Initializer/zerosFill1Variable_3/Adam/Initializer/zeros/shape_as_tensor'Variable_3/Adam/Initializer/zeros/Const*&
_output_shapes
:0@*
T0*
_class
loc:@Variable_3*

index_type0
²
Variable_3/Adam
VariableV2*
dtype0*&
_output_shapes
:0@*
shared_name *
_class
loc:@Variable_3*
	container *
shape:0@
Ķ
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:0@

Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*&
_output_shapes
:0@
«
3Variable_3/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_3*%
valueB"      0   @   *
dtype0*
_output_shapes
:

)Variable_3/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_3*
valueB
 *    *
dtype0*
_output_shapes
: 
ķ
#Variable_3/Adam_1/Initializer/zerosFill3Variable_3/Adam_1/Initializer/zeros/shape_as_tensor)Variable_3/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_3*

index_type0*&
_output_shapes
:0@
“
Variable_3/Adam_1
VariableV2*&
_output_shapes
:0@*
shared_name *
_class
loc:@Variable_3*
	container *
shape:0@*
dtype0
Ó
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:0@

Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*&
_output_shapes
:0@
©
1Variable_4/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_4*%
valueB"      @   @   *
dtype0*
_output_shapes
:

'Variable_4/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_4*
valueB
 *    *
dtype0*
_output_shapes
: 
ē
!Variable_4/Adam/Initializer/zerosFill1Variable_4/Adam/Initializer/zeros/shape_as_tensor'Variable_4/Adam/Initializer/zeros/Const*&
_output_shapes
:@@*
T0*
_class
loc:@Variable_4*

index_type0
²
Variable_4/Adam
VariableV2*
	container *
shape:@@*
dtype0*&
_output_shapes
:@@*
shared_name *
_class
loc:@Variable_4
Ķ
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0

Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*&
_output_shapes
:@@*
T0
«
3Variable_4/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_4*%
valueB"      @   @   *
dtype0*
_output_shapes
:

)Variable_4/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_4*
valueB
 *    *
dtype0*
_output_shapes
: 
ķ
#Variable_4/Adam_1/Initializer/zerosFill3Variable_4/Adam_1/Initializer/zeros/shape_as_tensor)Variable_4/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_4*

index_type0*&
_output_shapes
:@@
“
Variable_4/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape:@@*
dtype0*&
_output_shapes
:@@
Ó
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:@@

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*&
_output_shapes
:@@*
T0*
_class
loc:@Variable_4
”
1Variable_5/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_5*
valueB"  d   *
dtype0*
_output_shapes
:

'Variable_5/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_5*
valueB
 *    *
dtype0*
_output_shapes
: 
ą
!Variable_5/Adam/Initializer/zerosFill1Variable_5/Adam/Initializer/zeros/shape_as_tensor'Variable_5/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_5*

index_type0*
_output_shapes
:		d
¤
Variable_5/Adam
VariableV2*
dtype0*
_output_shapes
:		d*
shared_name *
_class
loc:@Variable_5*
	container *
shape:		d
Ę
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:		d*
use_locking(
z
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes
:		d*
T0*
_class
loc:@Variable_5
£
3Variable_5/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@Variable_5*
valueB"  d   

)Variable_5/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_5*
valueB
 *    *
dtype0*
_output_shapes
: 
ę
#Variable_5/Adam_1/Initializer/zerosFill3Variable_5/Adam_1/Initializer/zeros/shape_as_tensor)Variable_5/Adam_1/Initializer/zeros/Const*
_output_shapes
:		d*
T0*
_class
loc:@Variable_5*

index_type0
¦
Variable_5/Adam_1
VariableV2*
	container *
shape:		d*
dtype0*
_output_shapes
:		d*
shared_name *
_class
loc:@Variable_5
Ģ
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:		d
~
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes
:		d
”
1Variable_6/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_6*
valueB"d   2   *
dtype0*
_output_shapes
:

'Variable_6/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@Variable_6*
valueB
 *    *
dtype0
ß
!Variable_6/Adam/Initializer/zerosFill1Variable_6/Adam/Initializer/zeros/shape_as_tensor'Variable_6/Adam/Initializer/zeros/Const*
_class
loc:@Variable_6*

index_type0*
_output_shapes

:d2*
T0
¢
Variable_6/Adam
VariableV2*
	container *
shape
:d2*
dtype0*
_output_shapes

:d2*
shared_name *
_class
loc:@Variable_6
Å
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:d2
y
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
_output_shapes

:d2*
T0
£
3Variable_6/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_6*
valueB"d   2   *
dtype0*
_output_shapes
:

)Variable_6/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_6*
valueB
 *    *
dtype0*
_output_shapes
: 
å
#Variable_6/Adam_1/Initializer/zerosFill3Variable_6/Adam_1/Initializer/zeros/shape_as_tensor)Variable_6/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_6*

index_type0*
_output_shapes

:d2
¤
Variable_6/Adam_1
VariableV2*
shape
:d2*
dtype0*
_output_shapes

:d2*
shared_name *
_class
loc:@Variable_6*
	container 
Ė
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:d2*
use_locking(*
T0
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes

:d2

!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB2
*    *
dtype0*
_output_shapes

:2

¢
Variable_7/Adam
VariableV2*
dtype0*
_output_shapes

:2
*
shared_name *
_class
loc:@Variable_7*
	container *
shape
:2

Å
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
_output_shapes

:2
*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(
y
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes

:2


#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB2
*    *
dtype0*
_output_shapes

:2

¤
Variable_7/Adam_1
VariableV2*
shape
:2
*
dtype0*
_output_shapes

:2
*
shared_name *
_class
loc:@Variable_7*
	container 
Ė
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes

:2
*
use_locking(*
T0
}
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes

:2
*
T0*
_class
loc:@Variable_7

!Variable_8/Adam/Initializer/zerosConst*
_class
loc:@Variable_8*
valueB
*    *
dtype0*
_output_shapes

:

¢
Variable_8/Adam
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_8*
	container *
shape
:

Å
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes

:

y
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_class
loc:@Variable_8*
_output_shapes

:


#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*
_class
loc:@Variable_8*
valueB
*    
¤
Variable_8/Adam_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_8*
	container *
shape
:

Ė
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(
}
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
_output_shapes

:
*
T0*
_class
loc:@Variable_8

!Variable_9/Adam/Initializer/zerosConst*
_class
loc:@Variable_9*
valueB*    *
dtype0*
_output_shapes
:

Variable_9/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@Variable_9*
	container *
shape:
Į
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes
:
u
Variable_9/Adam/readIdentityVariable_9/Adam*
_output_shapes
:*
T0*
_class
loc:@Variable_9

#Variable_9/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_9*
valueB*    *
dtype0*
_output_shapes
:

Variable_9/Adam_1
VariableV2*
_class
loc:@Variable_9*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ē
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_class
loc:@Variable_9*
_output_shapes
:

"Variable_10/Adam/Initializer/zerosConst*
_class
loc:@Variable_10*
valueB$*    *
dtype0*
_output_shapes
:$

Variable_10/Adam
VariableV2*
dtype0*
_output_shapes
:$*
shared_name *
_class
loc:@Variable_10*
	container *
shape:$
Å
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:$
x
Variable_10/Adam/readIdentityVariable_10/Adam*
T0*
_class
loc:@Variable_10*
_output_shapes
:$

$Variable_10/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:$*
_class
loc:@Variable_10*
valueB$*    

Variable_10/Adam_1
VariableV2*
dtype0*
_output_shapes
:$*
shared_name *
_class
loc:@Variable_10*
	container *
shape:$
Ė
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:$*
use_locking(
|
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
_class
loc:@Variable_10*
_output_shapes
:$*
T0

"Variable_11/Adam/Initializer/zerosConst*
_class
loc:@Variable_11*
valueB0*    *
dtype0*
_output_shapes
:0

Variable_11/Adam
VariableV2*
dtype0*
_output_shapes
:0*
shared_name *
_class
loc:@Variable_11*
	container *
shape:0
Å
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
_output_shapes
:0*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(
x
Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_class
loc:@Variable_11*
_output_shapes
:0

$Variable_11/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_11*
valueB0*    *
dtype0*
_output_shapes
:0

Variable_11/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_11*
	container *
shape:0*
dtype0*
_output_shapes
:0
Ė
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
_output_shapes
:0*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(
|
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_class
loc:@Variable_11*
_output_shapes
:0

"Variable_12/Adam/Initializer/zerosConst*
_class
loc:@Variable_12*
valueB@*    *
dtype0*
_output_shapes
:@

Variable_12/Adam
VariableV2*
_class
loc:@Variable_12*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Å
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*
_output_shapes
:@
x
Variable_12/Adam/readIdentityVariable_12/Adam*
T0*
_class
loc:@Variable_12*
_output_shapes
:@

$Variable_12/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_12*
valueB@*    *
dtype0*
_output_shapes
:@

Variable_12/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_12*
	container *
shape:@*
dtype0*
_output_shapes
:@
Ė
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable_12
|
Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_class
loc:@Variable_12*
_output_shapes
:@

"Variable_13/Adam/Initializer/zerosConst*
_class
loc:@Variable_13*
valueB@*    *
dtype0*
_output_shapes
:@

Variable_13/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Variable_13*
	container 
Å
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes
:@
x
Variable_13/Adam/readIdentityVariable_13/Adam*
T0*
_class
loc:@Variable_13*
_output_shapes
:@

$Variable_13/Adam_1/Initializer/zerosConst*
_output_shapes
:@*
_class
loc:@Variable_13*
valueB@*    *
dtype0

Variable_13/Adam_1
VariableV2*
_class
loc:@Variable_13*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Ė
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes
:@
|
Variable_13/Adam_1/readIdentityVariable_13/Adam_1*
T0*
_class
loc:@Variable_13*
_output_shapes
:@

"Variable_14/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:d*
_class
loc:@Variable_14*
valueBd*    

Variable_14/Adam
VariableV2*
shared_name *
_class
loc:@Variable_14*
	container *
shape:d*
dtype0*
_output_shapes
:d
Å
Variable_14/Adam/AssignAssignVariable_14/Adam"Variable_14/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_14*
validate_shape(*
_output_shapes
:d*
use_locking(
x
Variable_14/Adam/readIdentityVariable_14/Adam*
T0*
_class
loc:@Variable_14*
_output_shapes
:d

$Variable_14/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_14*
valueBd*    *
dtype0*
_output_shapes
:d

Variable_14/Adam_1
VariableV2*
_output_shapes
:d*
shared_name *
_class
loc:@Variable_14*
	container *
shape:d*
dtype0
Ė
Variable_14/Adam_1/AssignAssignVariable_14/Adam_1$Variable_14/Adam_1/Initializer/zeros*
_class
loc:@Variable_14*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
|
Variable_14/Adam_1/readIdentityVariable_14/Adam_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_14

"Variable_15/Adam/Initializer/zerosConst*
_class
loc:@Variable_15*
valueB2*    *
dtype0*
_output_shapes
:2

Variable_15/Adam
VariableV2*
shared_name *
_class
loc:@Variable_15*
	container *
shape:2*
dtype0*
_output_shapes
:2
Å
Variable_15/Adam/AssignAssignVariable_15/Adam"Variable_15/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes
:2
x
Variable_15/Adam/readIdentityVariable_15/Adam*
T0*
_class
loc:@Variable_15*
_output_shapes
:2

$Variable_15/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_15*
valueB2*    *
dtype0*
_output_shapes
:2

Variable_15/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_15*
	container *
shape:2*
dtype0*
_output_shapes
:2
Ė
Variable_15/Adam_1/AssignAssignVariable_15/Adam_1$Variable_15/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes
:2*
use_locking(
|
Variable_15/Adam_1/readIdentityVariable_15/Adam_1*
_class
loc:@Variable_15*
_output_shapes
:2*
T0

"Variable_16/Adam/Initializer/zerosConst*
_class
loc:@Variable_16*
valueB
*    *
dtype0*
_output_shapes
:


Variable_16/Adam
VariableV2*
shared_name *
_class
loc:@Variable_16*
	container *
shape:
*
dtype0*
_output_shapes
:

Å
Variable_16/Adam/AssignAssignVariable_16/Adam"Variable_16/Adam/Initializer/zeros*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
x
Variable_16/Adam/readIdentityVariable_16/Adam*
_class
loc:@Variable_16*
_output_shapes
:
*
T0

$Variable_16/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_16*
valueB
*    *
dtype0*
_output_shapes
:


Variable_16/Adam_1
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@Variable_16*
	container *
shape:

Ė
Variable_16/Adam_1/AssignAssignVariable_16/Adam_1$Variable_16/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:

|
Variable_16/Adam_1/readIdentityVariable_16/Adam_1*
T0*
_class
loc:@Variable_16*
_output_shapes
:


"Variable_17/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@Variable_17*
valueB*    

Variable_17/Adam
VariableV2*
shared_name *
_class
loc:@Variable_17*
	container *
shape:*
dtype0*
_output_shapes
:
Å
Variable_17/Adam/AssignAssignVariable_17/Adam"Variable_17/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(
x
Variable_17/Adam/readIdentityVariable_17/Adam*
T0*
_class
loc:@Variable_17*
_output_shapes
:

$Variable_17/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_17*
valueB*    *
dtype0*
_output_shapes
:

Variable_17/Adam_1
VariableV2*
_class
loc:@Variable_17*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ė
Variable_17/Adam_1/AssignAssignVariable_17/Adam_1$Variable_17/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
|
Variable_17/Adam_1/readIdentityVariable_17/Adam_1*
T0*
_class
loc:@Variable_17*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *·Ń8
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *wĢ+2*
dtype0
»
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_17*
_class
loc:@Variable*
use_nesterov( *&
_output_shapes
:*
use_locking( *
T0
Å
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_15*
T0*
_class
loc:@Variable_1*
use_nesterov( *&
_output_shapes
:$*
use_locking( 
Å
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_13*
_class
loc:@Variable_2*
use_nesterov( *&
_output_shapes
:$0*
use_locking( *
T0
Å
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( *&
_output_shapes
:0@
Ä
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( *&
_output_shapes
:@@
½
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes
:		d
¼
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_locking( *
T0*
_class
loc:@Variable_6*
use_nesterov( *
_output_shapes

:d2
¼
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
_output_shapes

:2
*
use_locking( *
T0*
_class
loc:@Variable_7*
use_nesterov( 
¼
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_locking( *
T0*
_class
loc:@Variable_8*
use_nesterov( *
_output_shapes

:

¹
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_16*
use_locking( *
T0*
_class
loc:@Variable_9*
use_nesterov( *
_output_shapes
:
¾
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_14*
_class
loc:@Variable_10*
use_nesterov( *
_output_shapes
:$*
use_locking( *
T0
¾
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_12*
T0*
_class
loc:@Variable_11*
use_nesterov( *
_output_shapes
:0*
use_locking( 
¾
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
use_locking( *
T0*
_class
loc:@Variable_12*
use_nesterov( *
_output_shapes
:@
½
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@Variable_13*
use_nesterov( 
½
!Adam/update_Variable_14/ApplyAdam	ApplyAdamVariable_14Variable_14/AdamVariable_14/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
use_locking( *
T0*
_class
loc:@Variable_14*
use_nesterov( *
_output_shapes
:d
½
!Adam/update_Variable_15/ApplyAdam	ApplyAdamVariable_15Variable_15/AdamVariable_15/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_locking( *
T0*
_class
loc:@Variable_15*
use_nesterov( *
_output_shapes
:2
½
!Adam/update_Variable_16/ApplyAdam	ApplyAdamVariable_16Variable_16/AdamVariable_16/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_nesterov( *
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@Variable_16
»
!Adam/update_Variable_17/ApplyAdam	ApplyAdamVariable_17Variable_17/AdamVariable_17/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_locking( *
T0*
_class
loc:@Variable_17*
use_nesterov( *
_output_shapes
:
ē
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
é

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
¦
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam
Ų

initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_10/Adam/Assign^Variable_10/Adam_1/Assign^Variable_10/Assign^Variable_11/Adam/Assign^Variable_11/Adam_1/Assign^Variable_11/Assign^Variable_12/Adam/Assign^Variable_12/Adam_1/Assign^Variable_12/Assign^Variable_13/Adam/Assign^Variable_13/Adam_1/Assign^Variable_13/Assign^Variable_14/Adam/Assign^Variable_14/Adam_1/Assign^Variable_14/Assign^Variable_15/Adam/Assign^Variable_15/Adam_1/Assign^Variable_15/Assign^Variable_16/Adam/Assign^Variable_16/Adam_1/Assign^Variable_16/Assign^Variable_17/Adam/Assign^Variable_17/Adam_1/Assign^Variable_17/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^Variable_8/Adam/Assign^Variable_8/Adam_1/Assign^Variable_8/Assign^Variable_9/Adam/Assign^Variable_9/Adam_1/Assign^Variable_9/Assign^beta1_power/Assign^beta2_power/Assign
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_3b7508aeec7b4a05ada98791c41d5684/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
\
save/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
ī
save/SaveV2/tensor_namesConst*”
valueB8BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1BVariable_10BVariable_10/AdamBVariable_10/Adam_1BVariable_11BVariable_11/AdamBVariable_11/Adam_1BVariable_12BVariable_12/AdamBVariable_12/Adam_1BVariable_13BVariable_13/AdamBVariable_13/Adam_1BVariable_14BVariable_14/AdamBVariable_14/Adam_1BVariable_15BVariable_15/AdamBVariable_15/Adam_1BVariable_16BVariable_16/AdamBVariable_16/Adam_1BVariable_17BVariable_17/AdamBVariable_17/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1B
Variable_8BVariable_8/AdamBVariable_8/Adam_1B
Variable_9BVariable_9/AdamBVariable_9/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:8
Ō
save/SaveV2/shape_and_slicesConst*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:8
·
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1Variable_10Variable_10/AdamVariable_10/Adam_1Variable_11Variable_11/AdamVariable_11/Adam_1Variable_12Variable_12/AdamVariable_12/Adam_1Variable_13Variable_13/AdamVariable_13/Adam_1Variable_14Variable_14/AdamVariable_14/Adam_1Variable_15Variable_15/AdamVariable_15/Adam_1Variable_16Variable_16/AdamVariable_16/Adam_1Variable_17Variable_17/AdamVariable_17/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1
Variable_8Variable_8/AdamVariable_8/Adam_1
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_powerbeta2_power*F
dtypes<
:28

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*
T0*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
ń
save/RestoreV2/tensor_namesConst*”
valueB8BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1BVariable_10BVariable_10/AdamBVariable_10/Adam_1BVariable_11BVariable_11/AdamBVariable_11/Adam_1BVariable_12BVariable_12/AdamBVariable_12/Adam_1BVariable_13BVariable_13/AdamBVariable_13/Adam_1BVariable_14BVariable_14/AdamBVariable_14/Adam_1BVariable_15BVariable_15/AdamBVariable_15/Adam_1BVariable_16BVariable_16/AdamBVariable_16/Adam_1BVariable_17BVariable_17/AdamBVariable_17/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1B
Variable_8BVariable_8/AdamBVariable_8/Adam_1B
Variable_9BVariable_9/AdamBVariable_9/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:8
×
save/RestoreV2/shape_and_slicesConst*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:8
¦
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*ö
_output_shapesć
ą::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28
¦
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:
Æ
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:
±
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:
®
save/Assign_3Assign
Variable_1save/RestoreV2:3*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:$*
use_locking(
³
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*&
_output_shapes
:$*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
µ
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:$*
use_locking(
¤
save/Assign_6AssignVariable_10save/RestoreV2:6*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:$*
use_locking(
©
save/Assign_7AssignVariable_10/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:$
«
save/Assign_8AssignVariable_10/Adam_1save/RestoreV2:8*
_output_shapes
:$*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(
¤
save/Assign_9AssignVariable_11save/RestoreV2:9*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes
:0*
use_locking(*
T0
«
save/Assign_10AssignVariable_11/Adamsave/RestoreV2:10*
validate_shape(*
_output_shapes
:0*
use_locking(*
T0*
_class
loc:@Variable_11
­
save/Assign_11AssignVariable_11/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes
:0
¦
save/Assign_12AssignVariable_12save/RestoreV2:12*
T0*
_class
loc:@Variable_12*
validate_shape(*
_output_shapes
:@*
use_locking(
«
save/Assign_13AssignVariable_12/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*
_output_shapes
:@
­
save/Assign_14AssignVariable_12/Adam_1save/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*
_output_shapes
:@
¦
save/Assign_15AssignVariable_13save/RestoreV2:15*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes
:@*
use_locking(
«
save/Assign_16AssignVariable_13/Adamsave/RestoreV2:16*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
­
save/Assign_17AssignVariable_13/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable_13
¦
save/Assign_18AssignVariable_14save/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(*
_output_shapes
:d
«
save/Assign_19AssignVariable_14/Adamsave/RestoreV2:19*
_class
loc:@Variable_14*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
­
save/Assign_20AssignVariable_14/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(*
_output_shapes
:d
¦
save/Assign_21AssignVariable_15save/RestoreV2:21*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0*
_class
loc:@Variable_15
«
save/Assign_22AssignVariable_15/Adamsave/RestoreV2:22*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0*
_class
loc:@Variable_15
­
save/Assign_23AssignVariable_15/Adam_1save/RestoreV2:23*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes
:2*
use_locking(
¦
save/Assign_24AssignVariable_16save/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:

«
save/Assign_25AssignVariable_16/Adamsave/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:

­
save/Assign_26AssignVariable_16/Adam_1save/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:

¦
save/Assign_27AssignVariable_17save/RestoreV2:27*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
«
save/Assign_28AssignVariable_17/Adamsave/RestoreV2:28*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
­
save/Assign_29AssignVariable_17/Adam_1save/RestoreV2:29*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(
°
save/Assign_30Assign
Variable_2save/RestoreV2:30*&
_output_shapes
:$0*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
µ
save/Assign_31AssignVariable_2/Adamsave/RestoreV2:31*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:$0*
use_locking(
·
save/Assign_32AssignVariable_2/Adam_1save/RestoreV2:32*
validate_shape(*&
_output_shapes
:$0*
use_locking(*
T0*
_class
loc:@Variable_2
°
save/Assign_33Assign
Variable_3save/RestoreV2:33*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:0@*
use_locking(
µ
save/Assign_34AssignVariable_3/Adamsave/RestoreV2:34*&
_output_shapes
:0@*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
·
save/Assign_35AssignVariable_3/Adam_1save/RestoreV2:35*&
_output_shapes
:0@*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
°
save/Assign_36Assign
Variable_4save/RestoreV2:36*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:@@
µ
save/Assign_37AssignVariable_4/Adamsave/RestoreV2:37*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:@@
·
save/Assign_38AssignVariable_4/Adam_1save/RestoreV2:38*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:@@
©
save/Assign_39Assign
Variable_5save/RestoreV2:39*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:		d*
use_locking(*
T0
®
save/Assign_40AssignVariable_5/Adamsave/RestoreV2:40*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:		d*
use_locking(*
T0
°
save/Assign_41AssignVariable_5/Adam_1save/RestoreV2:41*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:		d*
use_locking(
Ø
save/Assign_42Assign
Variable_6save/RestoreV2:42*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:d2*
use_locking(*
T0
­
save/Assign_43AssignVariable_6/Adamsave/RestoreV2:43*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:d2
Æ
save/Assign_44AssignVariable_6/Adam_1save/RestoreV2:44*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:d2
Ø
save/Assign_45Assign
Variable_7save/RestoreV2:45*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes

:2

­
save/Assign_46AssignVariable_7/Adamsave/RestoreV2:46*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes

:2
*
use_locking(
Æ
save/Assign_47AssignVariable_7/Adam_1save/RestoreV2:47*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes

:2

Ø
save/Assign_48Assign
Variable_8save/RestoreV2:48*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
­
save/Assign_49AssignVariable_8/Adamsave/RestoreV2:49*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes

:

Æ
save/Assign_50AssignVariable_8/Adam_1save/RestoreV2:50*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes

:

¤
save/Assign_51Assign
Variable_9save/RestoreV2:51*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_9
©
save/Assign_52AssignVariable_9/Adamsave/RestoreV2:52*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(
«
save/Assign_53AssignVariable_9/Adam_1save/RestoreV2:53*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes
:

save/Assign_54Assignbeta1_powersave/RestoreV2:54*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 

save/Assign_55Assignbeta2_powersave/RestoreV2:55*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
Ę
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"
train_op

Adam"Ō+
	variablesĘ+Ć+
D

Variable:0Variable/AssignVariable/read:02truncated_normal:08
L
Variable_1:0Variable_1/AssignVariable_1/read:02truncated_normal_1:08
L
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_2:08
L
Variable_3:0Variable_3/AssignVariable_3/read:02truncated_normal_3:08
L
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_4:08
L
Variable_5:0Variable_5/AssignVariable_5/read:02truncated_normal_5:08
L
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_6:08
L
Variable_7:0Variable_7/AssignVariable_7/read:02truncated_normal_7:08
L
Variable_8:0Variable_8/AssignVariable_8/read:02truncated_normal_8:08
G
Variable_9:0Variable_9/AssignVariable_9/read:02random_normal:08
L
Variable_10:0Variable_10/AssignVariable_10/read:02random_normal_1:08
L
Variable_11:0Variable_11/AssignVariable_11/read:02random_normal_2:08
L
Variable_12:0Variable_12/AssignVariable_12/read:02random_normal_3:08
L
Variable_13:0Variable_13/AssignVariable_13/read:02random_normal_4:08
L
Variable_14:0Variable_14/AssignVariable_14/read:02random_normal_5:08
L
Variable_15:0Variable_15/AssignVariable_15/read:02random_normal_6:08
L
Variable_16:0Variable_16/AssignVariable_16/read:02random_normal_7:08
L
Variable_17:0Variable_17/AssignVariable_17/read:02random_normal_8:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
`
Variable/Adam:0Variable/Adam/AssignVariable/Adam/read:02!Variable/Adam/Initializer/zeros:0
h
Variable/Adam_1:0Variable/Adam_1/AssignVariable/Adam_1/read:02#Variable/Adam_1/Initializer/zeros:0
h
Variable_1/Adam:0Variable_1/Adam/AssignVariable_1/Adam/read:02#Variable_1/Adam/Initializer/zeros:0
p
Variable_1/Adam_1:0Variable_1/Adam_1/AssignVariable_1/Adam_1/read:02%Variable_1/Adam_1/Initializer/zeros:0
h
Variable_2/Adam:0Variable_2/Adam/AssignVariable_2/Adam/read:02#Variable_2/Adam/Initializer/zeros:0
p
Variable_2/Adam_1:0Variable_2/Adam_1/AssignVariable_2/Adam_1/read:02%Variable_2/Adam_1/Initializer/zeros:0
h
Variable_3/Adam:0Variable_3/Adam/AssignVariable_3/Adam/read:02#Variable_3/Adam/Initializer/zeros:0
p
Variable_3/Adam_1:0Variable_3/Adam_1/AssignVariable_3/Adam_1/read:02%Variable_3/Adam_1/Initializer/zeros:0
h
Variable_4/Adam:0Variable_4/Adam/AssignVariable_4/Adam/read:02#Variable_4/Adam/Initializer/zeros:0
p
Variable_4/Adam_1:0Variable_4/Adam_1/AssignVariable_4/Adam_1/read:02%Variable_4/Adam_1/Initializer/zeros:0
h
Variable_5/Adam:0Variable_5/Adam/AssignVariable_5/Adam/read:02#Variable_5/Adam/Initializer/zeros:0
p
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0
h
Variable_6/Adam:0Variable_6/Adam/AssignVariable_6/Adam/read:02#Variable_6/Adam/Initializer/zeros:0
p
Variable_6/Adam_1:0Variable_6/Adam_1/AssignVariable_6/Adam_1/read:02%Variable_6/Adam_1/Initializer/zeros:0
h
Variable_7/Adam:0Variable_7/Adam/AssignVariable_7/Adam/read:02#Variable_7/Adam/Initializer/zeros:0
p
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0
h
Variable_8/Adam:0Variable_8/Adam/AssignVariable_8/Adam/read:02#Variable_8/Adam/Initializer/zeros:0
p
Variable_8/Adam_1:0Variable_8/Adam_1/AssignVariable_8/Adam_1/read:02%Variable_8/Adam_1/Initializer/zeros:0
h
Variable_9/Adam:0Variable_9/Adam/AssignVariable_9/Adam/read:02#Variable_9/Adam/Initializer/zeros:0
p
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0
l
Variable_10/Adam:0Variable_10/Adam/AssignVariable_10/Adam/read:02$Variable_10/Adam/Initializer/zeros:0
t
Variable_10/Adam_1:0Variable_10/Adam_1/AssignVariable_10/Adam_1/read:02&Variable_10/Adam_1/Initializer/zeros:0
l
Variable_11/Adam:0Variable_11/Adam/AssignVariable_11/Adam/read:02$Variable_11/Adam/Initializer/zeros:0
t
Variable_11/Adam_1:0Variable_11/Adam_1/AssignVariable_11/Adam_1/read:02&Variable_11/Adam_1/Initializer/zeros:0
l
Variable_12/Adam:0Variable_12/Adam/AssignVariable_12/Adam/read:02$Variable_12/Adam/Initializer/zeros:0
t
Variable_12/Adam_1:0Variable_12/Adam_1/AssignVariable_12/Adam_1/read:02&Variable_12/Adam_1/Initializer/zeros:0
l
Variable_13/Adam:0Variable_13/Adam/AssignVariable_13/Adam/read:02$Variable_13/Adam/Initializer/zeros:0
t
Variable_13/Adam_1:0Variable_13/Adam_1/AssignVariable_13/Adam_1/read:02&Variable_13/Adam_1/Initializer/zeros:0
l
Variable_14/Adam:0Variable_14/Adam/AssignVariable_14/Adam/read:02$Variable_14/Adam/Initializer/zeros:0
t
Variable_14/Adam_1:0Variable_14/Adam_1/AssignVariable_14/Adam_1/read:02&Variable_14/Adam_1/Initializer/zeros:0
l
Variable_15/Adam:0Variable_15/Adam/AssignVariable_15/Adam/read:02$Variable_15/Adam/Initializer/zeros:0
t
Variable_15/Adam_1:0Variable_15/Adam_1/AssignVariable_15/Adam_1/read:02&Variable_15/Adam_1/Initializer/zeros:0
l
Variable_16/Adam:0Variable_16/Adam/AssignVariable_16/Adam/read:02$Variable_16/Adam/Initializer/zeros:0
t
Variable_16/Adam_1:0Variable_16/Adam_1/AssignVariable_16/Adam_1/read:02&Variable_16/Adam_1/Initializer/zeros:0
l
Variable_17/Adam:0Variable_17/Adam/AssignVariable_17/Adam/read:02$Variable_17/Adam/Initializer/zeros:0
t
Variable_17/Adam_1:0Variable_17/Adam_1/AssignVariable_17/Adam_1/read:02&Variable_17/Adam_1/Initializer/zeros:0"
trainable_variablesņ
ļ

D

Variable:0Variable/AssignVariable/read:02truncated_normal:08
L
Variable_1:0Variable_1/AssignVariable_1/read:02truncated_normal_1:08
L
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_2:08
L
Variable_3:0Variable_3/AssignVariable_3/read:02truncated_normal_3:08
L
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_4:08
L
Variable_5:0Variable_5/AssignVariable_5/read:02truncated_normal_5:08
L
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_6:08
L
Variable_7:0Variable_7/AssignVariable_7/read:02truncated_normal_7:08
L
Variable_8:0Variable_8/AssignVariable_8/read:02truncated_normal_8:08
G
Variable_9:0Variable_9/AssignVariable_9/read:02random_normal:08
L
Variable_10:0Variable_10/AssignVariable_10/read:02random_normal_1:08
L
Variable_11:0Variable_11/AssignVariable_11/read:02random_normal_2:08
L
Variable_12:0Variable_12/AssignVariable_12/read:02random_normal_3:08
L
Variable_13:0Variable_13/AssignVariable_13/read:02random_normal_4:08
L
Variable_14:0Variable_14/AssignVariable_14/read:02random_normal_5:08
L
Variable_15:0Variable_15/AssignVariable_15/read:02random_normal_6:08
L
Variable_16:0Variable_16/AssignVariable_16/read:02random_normal_7:08
L
Variable_17:0Variable_17/AssignVariable_17/read:02random_normal_8:08*
serving_default|
4
myInput)
	myInput:0’’’’’’’’’BČ(
myOutput
Mul:0’’’’’’’’’tensorflow/serving/predict