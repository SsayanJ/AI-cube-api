üÒ
Ç
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Ç
£
#dueling_deep_q_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#dueling_deep_q_network/dense/kernel

7dueling_deep_q_network/dense/kernel/Read/ReadVariableOpReadVariableOp#dueling_deep_q_network/dense/kernel*
_output_shapes
:	*
dtype0

!dueling_deep_q_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!dueling_deep_q_network/dense/bias

5dueling_deep_q_network/dense/bias/Read/ReadVariableOpReadVariableOp!dueling_deep_q_network/dense/bias*
_output_shapes	
:*
dtype0
¨
%dueling_deep_q_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%dueling_deep_q_network/dense_1/kernel
¡
9dueling_deep_q_network/dense_1/kernel/Read/ReadVariableOpReadVariableOp%dueling_deep_q_network/dense_1/kernel* 
_output_shapes
:
*
dtype0

#dueling_deep_q_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#dueling_deep_q_network/dense_1/bias

7dueling_deep_q_network/dense_1/bias/Read/ReadVariableOpReadVariableOp#dueling_deep_q_network/dense_1/bias*
_output_shapes	
:*
dtype0
§
%dueling_deep_q_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%dueling_deep_q_network/dense_2/kernel
 
9dueling_deep_q_network/dense_2/kernel/Read/ReadVariableOpReadVariableOp%dueling_deep_q_network/dense_2/kernel*
_output_shapes
:	*
dtype0

#dueling_deep_q_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#dueling_deep_q_network/dense_2/bias

7dueling_deep_q_network/dense_2/bias/Read/ReadVariableOpReadVariableOp#dueling_deep_q_network/dense_2/bias*
_output_shapes
:*
dtype0
§
%dueling_deep_q_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	;*6
shared_name'%dueling_deep_q_network/dense_3/kernel
 
9dueling_deep_q_network/dense_3/kernel/Read/ReadVariableOpReadVariableOp%dueling_deep_q_network/dense_3/kernel*
_output_shapes
:	;*
dtype0

#dueling_deep_q_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:;*4
shared_name%#dueling_deep_q_network/dense_3/bias

7dueling_deep_q_network/dense_3/bias/Read/ReadVariableOpReadVariableOp#dueling_deep_q_network/dense_3/bias*
_output_shapes
:;*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
±
*Adam/dueling_deep_q_network/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/dueling_deep_q_network/dense/kernel/m
ª
>Adam/dueling_deep_q_network/dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense/kernel/m*
_output_shapes
:	*
dtype0
©
(Adam/dueling_deep_q_network/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/dueling_deep_q_network/dense/bias/m
¢
<Adam/dueling_deep_q_network/dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/dueling_deep_q_network/dense/bias/m*
_output_shapes	
:*
dtype0
¶
,Adam/dueling_deep_q_network/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/dueling_deep_q_network/dense_1/kernel/m
¯
@Adam/dueling_deep_q_network/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_1/kernel/m* 
_output_shapes
:
*
dtype0
­
*Adam/dueling_deep_q_network/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_1/bias/m
¦
>Adam/dueling_deep_q_network/dense_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_1/bias/m*
_output_shapes	
:*
dtype0
µ
,Adam/dueling_deep_q_network/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/dueling_deep_q_network/dense_2/kernel/m
®
@Adam/dueling_deep_q_network/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_2/kernel/m*
_output_shapes
:	*
dtype0
¬
*Adam/dueling_deep_q_network/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_2/bias/m
¥
>Adam/dueling_deep_q_network/dense_2/bias/m/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_2/bias/m*
_output_shapes
:*
dtype0
µ
,Adam/dueling_deep_q_network/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	;*=
shared_name.,Adam/dueling_deep_q_network/dense_3/kernel/m
®
@Adam/dueling_deep_q_network/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_3/kernel/m*
_output_shapes
:	;*
dtype0
¬
*Adam/dueling_deep_q_network/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:;*;
shared_name,*Adam/dueling_deep_q_network/dense_3/bias/m
¥
>Adam/dueling_deep_q_network/dense_3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_3/bias/m*
_output_shapes
:;*
dtype0
±
*Adam/dueling_deep_q_network/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/dueling_deep_q_network/dense/kernel/v
ª
>Adam/dueling_deep_q_network/dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense/kernel/v*
_output_shapes
:	*
dtype0
©
(Adam/dueling_deep_q_network/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/dueling_deep_q_network/dense/bias/v
¢
<Adam/dueling_deep_q_network/dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/dueling_deep_q_network/dense/bias/v*
_output_shapes	
:*
dtype0
¶
,Adam/dueling_deep_q_network/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/dueling_deep_q_network/dense_1/kernel/v
¯
@Adam/dueling_deep_q_network/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_1/kernel/v* 
_output_shapes
:
*
dtype0
­
*Adam/dueling_deep_q_network/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_1/bias/v
¦
>Adam/dueling_deep_q_network/dense_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_1/bias/v*
_output_shapes	
:*
dtype0
µ
,Adam/dueling_deep_q_network/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/dueling_deep_q_network/dense_2/kernel/v
®
@Adam/dueling_deep_q_network/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_2/kernel/v*
_output_shapes
:	*
dtype0
¬
*Adam/dueling_deep_q_network/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/dueling_deep_q_network/dense_2/bias/v
¥
>Adam/dueling_deep_q_network/dense_2/bias/v/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_2/bias/v*
_output_shapes
:*
dtype0
µ
,Adam/dueling_deep_q_network/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	;*=
shared_name.,Adam/dueling_deep_q_network/dense_3/kernel/v
®
@Adam/dueling_deep_q_network/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/dueling_deep_q_network/dense_3/kernel/v*
_output_shapes
:	;*
dtype0
¬
*Adam/dueling_deep_q_network/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:;*;
shared_name,*Adam/dueling_deep_q_network/dense_3/bias/v
¥
>Adam/dueling_deep_q_network/dense_3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/dueling_deep_q_network/dense_3/bias/v*
_output_shapes
:;*
dtype0

NoOpNoOp
ü+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*·+
value­+Bª+ B£+


dense1

dense2
V
A
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
Ð
#iter

$beta_1

%beta_2
	&decay
'learning_ratemFmGmHmImJmKmLmMvNvOvPvQvRvSvTvU
 
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
­
(non_trainable_variables
regularization_losses
)metrics
*layer_regularization_losses
	variables

+layers
trainable_variables
,layer_metrics
 
a_
VARIABLE_VALUE#dueling_deep_q_network/dense/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!dueling_deep_q_network/dense/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
-non_trainable_variables
regularization_losses
.metrics
/layer_regularization_losses
	variables

0layers
trainable_variables
1layer_metrics
ca
VARIABLE_VALUE%dueling_deep_q_network/dense_1/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#dueling_deep_q_network/dense_1/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
2non_trainable_variables
regularization_losses
3metrics
4layer_regularization_losses
	variables

5layers
trainable_variables
6layer_metrics
^\
VARIABLE_VALUE%dueling_deep_q_network/dense_2/kernel#V/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE#dueling_deep_q_network/dense_2/bias!V/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
7non_trainable_variables
regularization_losses
8metrics
9layer_regularization_losses
	variables

:layers
trainable_variables
;layer_metrics
^\
VARIABLE_VALUE%dueling_deep_q_network/dense_3/kernel#A/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE#dueling_deep_q_network/dense_3/bias!A/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
<non_trainable_variables
regularization_losses
=metrics
>layer_regularization_losses
 	variables

?layers
!trainable_variables
@layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

A0
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Btotal
	Ccount
D	variables
E	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

D	variables

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE(Adam/dueling_deep_q_network/dense/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_1/kernel/mDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_1/bias/mBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_2/kernel/m?V/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_2/bias/m=V/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_3/kernel/m?A/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_3/bias/m=A/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE(Adam/dueling_deep_q_network/dense/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_1/kernel/vDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_1/bias/vBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_2/kernel/v?V/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_2/bias/v=V/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/dueling_deep_q_network/dense_3/kernel/v?A/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE*Adam/dueling_deep_q_network/dense_3/bias/v=A/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ð
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#dueling_deep_q_network/dense/kernel!dueling_deep_q_network/dense/bias%dueling_deep_q_network/dense_1/kernel#dueling_deep_q_network/dense_1/bias%dueling_deep_q_network/dense_2/kernel#dueling_deep_q_network/dense_2/bias%dueling_deep_q_network/dense_3/kernel#dueling_deep_q_network/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_27769113
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ï
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7dueling_deep_q_network/dense/kernel/Read/ReadVariableOp5dueling_deep_q_network/dense/bias/Read/ReadVariableOp9dueling_deep_q_network/dense_1/kernel/Read/ReadVariableOp7dueling_deep_q_network/dense_1/bias/Read/ReadVariableOp9dueling_deep_q_network/dense_2/kernel/Read/ReadVariableOp7dueling_deep_q_network/dense_2/bias/Read/ReadVariableOp9dueling_deep_q_network/dense_3/kernel/Read/ReadVariableOp7dueling_deep_q_network/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense/kernel/m/Read/ReadVariableOp<Adam/dueling_deep_q_network/dense/bias/m/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_1/kernel/m/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_1/bias/m/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_2/kernel/m/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_2/bias/m/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_3/kernel/m/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_3/bias/m/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense/kernel/v/Read/ReadVariableOp<Adam/dueling_deep_q_network/dense/bias/v/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_1/kernel/v/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_1/bias/v/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_2/kernel/v/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_2/bias/v/Read/ReadVariableOp@Adam/dueling_deep_q_network/dense_3/kernel/v/Read/ReadVariableOp>Adam/dueling_deep_q_network/dense_3/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_27769307
þ

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#dueling_deep_q_network/dense/kernel!dueling_deep_q_network/dense/bias%dueling_deep_q_network/dense_1/kernel#dueling_deep_q_network/dense_1/bias%dueling_deep_q_network/dense_2/kernel#dueling_deep_q_network/dense_2/bias%dueling_deep_q_network/dense_3/kernel#dueling_deep_q_network/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount*Adam/dueling_deep_q_network/dense/kernel/m(Adam/dueling_deep_q_network/dense/bias/m,Adam/dueling_deep_q_network/dense_1/kernel/m*Adam/dueling_deep_q_network/dense_1/bias/m,Adam/dueling_deep_q_network/dense_2/kernel/m*Adam/dueling_deep_q_network/dense_2/bias/m,Adam/dueling_deep_q_network/dense_3/kernel/m*Adam/dueling_deep_q_network/dense_3/bias/m*Adam/dueling_deep_q_network/dense/kernel/v(Adam/dueling_deep_q_network/dense/bias/v,Adam/dueling_deep_q_network/dense_1/kernel/v*Adam/dueling_deep_q_network/dense_1/bias/v,Adam/dueling_deep_q_network/dense_2/kernel/v*Adam/dueling_deep_q_network/dense_2/bias/v,Adam/dueling_deep_q_network/dense_3/kernel/v*Adam/dueling_deep_q_network/dense_3/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_27769410·®
Á	
Ð
9__inference_dueling_deep_q_network_layer_call_fn_27769044
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	;
	unknown_6:;
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_277690222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
 

*__inference_dense_3_layer_call_fn_27769181

inputs
unknown:	;
	unknown_0:;
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_277690112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

T__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_27769022
input_1!
dense_27768963:	
dense_27768965:	$
dense_1_27768980:

dense_1_27768982:	#
dense_2_27768996:	
dense_2_27768998:#
dense_3_27769012:	;
dense_3_27769014:;
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_27768963dense_27768965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_277689622
dense/StatefulPartitionedCall¶
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_27768980dense_1_27768982*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_277689792!
dense_1/StatefulPartitionedCall·
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_27768996dense_2_27768998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_277689952!
dense_2/StatefulPartitionedCall·
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_27769012dense_3_27769014*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_277690112!
dense_3/StatefulPartitionedCallr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices¢
MeanMean(dense_3/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Mean|
subSub(dense_3/StatefulPartitionedCall:output:0Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2
subx
addAddV2(dense_2/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2
addá
IdentityIdentityadd:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
 =
ú
#__inference__wrapped_model_27768947
input_1N
;dueling_deep_q_network_dense_matmul_readvariableop_resource:	K
<dueling_deep_q_network_dense_biasadd_readvariableop_resource:	Q
=dueling_deep_q_network_dense_1_matmul_readvariableop_resource:
M
>dueling_deep_q_network_dense_1_biasadd_readvariableop_resource:	P
=dueling_deep_q_network_dense_2_matmul_readvariableop_resource:	L
>dueling_deep_q_network_dense_2_biasadd_readvariableop_resource:P
=dueling_deep_q_network_dense_3_matmul_readvariableop_resource:	;L
>dueling_deep_q_network_dense_3_biasadd_readvariableop_resource:;
identity¢3dueling_deep_q_network/dense/BiasAdd/ReadVariableOp¢2dueling_deep_q_network/dense/MatMul/ReadVariableOp¢5dueling_deep_q_network/dense_1/BiasAdd/ReadVariableOp¢4dueling_deep_q_network/dense_1/MatMul/ReadVariableOp¢5dueling_deep_q_network/dense_2/BiasAdd/ReadVariableOp¢4dueling_deep_q_network/dense_2/MatMul/ReadVariableOp¢5dueling_deep_q_network/dense_3/BiasAdd/ReadVariableOp¢4dueling_deep_q_network/dense_3/MatMul/ReadVariableOpå
2dueling_deep_q_network/dense/MatMul/ReadVariableOpReadVariableOp;dueling_deep_q_network_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2dueling_deep_q_network/dense/MatMul/ReadVariableOpÌ
#dueling_deep_q_network/dense/MatMulMatMulinput_1:dueling_deep_q_network/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#dueling_deep_q_network/dense/MatMulä
3dueling_deep_q_network/dense/BiasAdd/ReadVariableOpReadVariableOp<dueling_deep_q_network_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3dueling_deep_q_network/dense/BiasAdd/ReadVariableOpö
$dueling_deep_q_network/dense/BiasAddBiasAdd-dueling_deep_q_network/dense/MatMul:product:0;dueling_deep_q_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$dueling_deep_q_network/dense/BiasAdd°
!dueling_deep_q_network/dense/ReluRelu-dueling_deep_q_network/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!dueling_deep_q_network/dense/Reluì
4dueling_deep_q_network/dense_1/MatMul/ReadVariableOpReadVariableOp=dueling_deep_q_network_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype026
4dueling_deep_q_network/dense_1/MatMul/ReadVariableOpú
%dueling_deep_q_network/dense_1/MatMulMatMul/dueling_deep_q_network/dense/Relu:activations:0<dueling_deep_q_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%dueling_deep_q_network/dense_1/MatMulê
5dueling_deep_q_network/dense_1/BiasAdd/ReadVariableOpReadVariableOp>dueling_deep_q_network_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5dueling_deep_q_network/dense_1/BiasAdd/ReadVariableOpþ
&dueling_deep_q_network/dense_1/BiasAddBiasAdd/dueling_deep_q_network/dense_1/MatMul:product:0=dueling_deep_q_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&dueling_deep_q_network/dense_1/BiasAdd¶
#dueling_deep_q_network/dense_1/ReluRelu/dueling_deep_q_network/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#dueling_deep_q_network/dense_1/Reluë
4dueling_deep_q_network/dense_2/MatMul/ReadVariableOpReadVariableOp=dueling_deep_q_network_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype026
4dueling_deep_q_network/dense_2/MatMul/ReadVariableOpû
%dueling_deep_q_network/dense_2/MatMulMatMul1dueling_deep_q_network/dense_1/Relu:activations:0<dueling_deep_q_network/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%dueling_deep_q_network/dense_2/MatMulé
5dueling_deep_q_network/dense_2/BiasAdd/ReadVariableOpReadVariableOp>dueling_deep_q_network_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5dueling_deep_q_network/dense_2/BiasAdd/ReadVariableOpý
&dueling_deep_q_network/dense_2/BiasAddBiasAdd/dueling_deep_q_network/dense_2/MatMul:product:0=dueling_deep_q_network/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&dueling_deep_q_network/dense_2/BiasAddë
4dueling_deep_q_network/dense_3/MatMul/ReadVariableOpReadVariableOp=dueling_deep_q_network_dense_3_matmul_readvariableop_resource*
_output_shapes
:	;*
dtype026
4dueling_deep_q_network/dense_3/MatMul/ReadVariableOpû
%dueling_deep_q_network/dense_3/MatMulMatMul1dueling_deep_q_network/dense_1/Relu:activations:0<dueling_deep_q_network/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2'
%dueling_deep_q_network/dense_3/MatMulé
5dueling_deep_q_network/dense_3/BiasAdd/ReadVariableOpReadVariableOp>dueling_deep_q_network_dense_3_biasadd_readvariableop_resource*
_output_shapes
:;*
dtype027
5dueling_deep_q_network/dense_3/BiasAdd/ReadVariableOpý
&dueling_deep_q_network/dense_3/BiasAddBiasAdd/dueling_deep_q_network/dense_3/MatMul:product:0=dueling_deep_q_network/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2(
&dueling_deep_q_network/dense_3/BiasAdd 
-dueling_deep_q_network/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-dueling_deep_q_network/Mean/reduction_indicesî
dueling_deep_q_network/MeanMean/dueling_deep_q_network/dense_3/BiasAdd:output:06dueling_deep_q_network/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
dueling_deep_q_network/MeanÈ
dueling_deep_q_network/subSub/dueling_deep_q_network/dense_3/BiasAdd:output:0$dueling_deep_q_network/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2
dueling_deep_q_network/subÄ
dueling_deep_q_network/addAddV2/dueling_deep_q_network/dense_2/BiasAdd:output:0dueling_deep_q_network/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2
dueling_deep_q_network/addª
IdentityIdentitydueling_deep_q_network/add:z:04^dueling_deep_q_network/dense/BiasAdd/ReadVariableOp3^dueling_deep_q_network/dense/MatMul/ReadVariableOp6^dueling_deep_q_network/dense_1/BiasAdd/ReadVariableOp5^dueling_deep_q_network/dense_1/MatMul/ReadVariableOp6^dueling_deep_q_network/dense_2/BiasAdd/ReadVariableOp5^dueling_deep_q_network/dense_2/MatMul/ReadVariableOp6^dueling_deep_q_network/dense_3/BiasAdd/ReadVariableOp5^dueling_deep_q_network/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2j
3dueling_deep_q_network/dense/BiasAdd/ReadVariableOp3dueling_deep_q_network/dense/BiasAdd/ReadVariableOp2h
2dueling_deep_q_network/dense/MatMul/ReadVariableOp2dueling_deep_q_network/dense/MatMul/ReadVariableOp2n
5dueling_deep_q_network/dense_1/BiasAdd/ReadVariableOp5dueling_deep_q_network/dense_1/BiasAdd/ReadVariableOp2l
4dueling_deep_q_network/dense_1/MatMul/ReadVariableOp4dueling_deep_q_network/dense_1/MatMul/ReadVariableOp2n
5dueling_deep_q_network/dense_2/BiasAdd/ReadVariableOp5dueling_deep_q_network/dense_2/BiasAdd/ReadVariableOp2l
4dueling_deep_q_network/dense_2/MatMul/ReadVariableOp4dueling_deep_q_network/dense_2/MatMul/ReadVariableOp2n
5dueling_deep_q_network/dense_3/BiasAdd/ReadVariableOp5dueling_deep_q_network/dense_3/BiasAdd/ReadVariableOp2l
4dueling_deep_q_network/dense_3/MatMul/ReadVariableOp4dueling_deep_q_network/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
 

*__inference_dense_2_layer_call_fn_27769162

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_277689952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

ö
C__inference_dense_layer_call_and_return_conditional_losses_27768962

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

*__inference_dense_1_layer_call_fn_27769142

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_277689792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ	
÷
E__inference_dense_3_layer_call_and_return_conditional_losses_27769191

inputs1
matmul_readvariableop_resource:	;-
biasadd_readvariableop_resource:;
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	;*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:;*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ	
÷
E__inference_dense_2_layer_call_and_return_conditional_losses_27768995

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹

ù
E__inference_dense_1_layer_call_and_return_conditional_losses_27768979

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_dense_layer_call_fn_27769122

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_277689622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

ö
C__inference_dense_layer_call_and_return_conditional_losses_27769133

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹

ù
E__inference_dense_1_layer_call_and_return_conditional_losses_27769153

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
ë
$__inference__traced_restore_27769410
file_prefixG
4assignvariableop_dueling_deep_q_network_dense_kernel:	C
4assignvariableop_1_dueling_deep_q_network_dense_bias:	L
8assignvariableop_2_dueling_deep_q_network_dense_1_kernel:
E
6assignvariableop_3_dueling_deep_q_network_dense_1_bias:	K
8assignvariableop_4_dueling_deep_q_network_dense_2_kernel:	D
6assignvariableop_5_dueling_deep_q_network_dense_2_bias:K
8assignvariableop_6_dueling_deep_q_network_dense_3_kernel:	;D
6assignvariableop_7_dueling_deep_q_network_dense_3_bias:;&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: Q
>assignvariableop_15_adam_dueling_deep_q_network_dense_kernel_m:	K
<assignvariableop_16_adam_dueling_deep_q_network_dense_bias_m:	T
@assignvariableop_17_adam_dueling_deep_q_network_dense_1_kernel_m:
M
>assignvariableop_18_adam_dueling_deep_q_network_dense_1_bias_m:	S
@assignvariableop_19_adam_dueling_deep_q_network_dense_2_kernel_m:	L
>assignvariableop_20_adam_dueling_deep_q_network_dense_2_bias_m:S
@assignvariableop_21_adam_dueling_deep_q_network_dense_3_kernel_m:	;L
>assignvariableop_22_adam_dueling_deep_q_network_dense_3_bias_m:;Q
>assignvariableop_23_adam_dueling_deep_q_network_dense_kernel_v:	K
<assignvariableop_24_adam_dueling_deep_q_network_dense_bias_v:	T
@assignvariableop_25_adam_dueling_deep_q_network_dense_1_kernel_v:
M
>assignvariableop_26_adam_dueling_deep_q_network_dense_1_bias_v:	S
@assignvariableop_27_adam_dueling_deep_q_network_dense_2_kernel_v:	L
>assignvariableop_28_adam_dueling_deep_q_network_dense_2_bias_v:S
@assignvariableop_29_adam_dueling_deep_q_network_dense_3_kernel_v:	;L
>assignvariableop_30_adam_dueling_deep_q_network_dense_3_bias_v:;
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ô
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*à
valueÖBÓ B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB#V/kernel/.ATTRIBUTES/VARIABLE_VALUEB!V/bias/.ATTRIBUTES/VARIABLE_VALUEB#A/kernel/.ATTRIBUTES/VARIABLE_VALUEB!A/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?V/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=V/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?A/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=A/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?V/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=V/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?A/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=A/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity³
AssignVariableOpAssignVariableOp4assignvariableop_dueling_deep_q_network_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¹
AssignVariableOp_1AssignVariableOp4assignvariableop_1_dueling_deep_q_network_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2½
AssignVariableOp_2AssignVariableOp8assignvariableop_2_dueling_deep_q_network_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3»
AssignVariableOp_3AssignVariableOp6assignvariableop_3_dueling_deep_q_network_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4½
AssignVariableOp_4AssignVariableOp8assignvariableop_4_dueling_deep_q_network_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5»
AssignVariableOp_5AssignVariableOp6assignvariableop_5_dueling_deep_q_network_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6½
AssignVariableOp_6AssignVariableOp8assignvariableop_6_dueling_deep_q_network_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7»
AssignVariableOp_7AssignVariableOp6assignvariableop_7_dueling_deep_q_network_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8¡
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10§
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¦
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12®
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¡
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¡
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Æ
AssignVariableOp_15AssignVariableOp>assignvariableop_15_adam_dueling_deep_q_network_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ä
AssignVariableOp_16AssignVariableOp<assignvariableop_16_adam_dueling_deep_q_network_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17È
AssignVariableOp_17AssignVariableOp@assignvariableop_17_adam_dueling_deep_q_network_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Æ
AssignVariableOp_18AssignVariableOp>assignvariableop_18_adam_dueling_deep_q_network_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19È
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_dueling_deep_q_network_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Æ
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_dueling_deep_q_network_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21È
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_dueling_deep_q_network_dense_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Æ
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_dueling_deep_q_network_dense_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Æ
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_dueling_deep_q_network_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ä
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_dueling_deep_q_network_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25È
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_dueling_deep_q_network_dense_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Æ
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_dueling_deep_q_network_dense_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27È
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_dueling_deep_q_network_dense_2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Æ
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_dueling_deep_q_network_dense_2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29È
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_dueling_deep_q_network_dense_3_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Æ
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_dueling_deep_q_network_dense_3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31û
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Õ	
÷
E__inference_dense_2_layer_call_and_return_conditional_losses_27769172

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦J
ö
!__inference__traced_save_27769307
file_prefixB
>savev2_dueling_deep_q_network_dense_kernel_read_readvariableop@
<savev2_dueling_deep_q_network_dense_bias_read_readvariableopD
@savev2_dueling_deep_q_network_dense_1_kernel_read_readvariableopB
>savev2_dueling_deep_q_network_dense_1_bias_read_readvariableopD
@savev2_dueling_deep_q_network_dense_2_kernel_read_readvariableopB
>savev2_dueling_deep_q_network_dense_2_bias_read_readvariableopD
@savev2_dueling_deep_q_network_dense_3_kernel_read_readvariableopB
>savev2_dueling_deep_q_network_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_kernel_m_read_readvariableopG
Csavev2_adam_dueling_deep_q_network_dense_bias_m_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_1_kernel_m_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_1_bias_m_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_2_kernel_m_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_2_bias_m_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_3_kernel_m_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_3_bias_m_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_kernel_v_read_readvariableopG
Csavev2_adam_dueling_deep_q_network_dense_bias_v_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_1_kernel_v_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_1_bias_v_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_2_kernel_v_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_2_bias_v_read_readvariableopK
Gsavev2_adam_dueling_deep_q_network_dense_3_kernel_v_read_readvariableopI
Esavev2_adam_dueling_deep_q_network_dense_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÎ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*à
valueÖBÓ B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB#V/kernel/.ATTRIBUTES/VARIABLE_VALUEB!V/bias/.ATTRIBUTES/VARIABLE_VALUEB#A/kernel/.ATTRIBUTES/VARIABLE_VALUEB!A/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?V/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=V/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?A/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=A/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?V/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=V/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?A/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=A/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÈ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_sliceså
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_dueling_deep_q_network_dense_kernel_read_readvariableop<savev2_dueling_deep_q_network_dense_bias_read_readvariableop@savev2_dueling_deep_q_network_dense_1_kernel_read_readvariableop>savev2_dueling_deep_q_network_dense_1_bias_read_readvariableop@savev2_dueling_deep_q_network_dense_2_kernel_read_readvariableop>savev2_dueling_deep_q_network_dense_2_bias_read_readvariableop@savev2_dueling_deep_q_network_dense_3_kernel_read_readvariableop>savev2_dueling_deep_q_network_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_kernel_m_read_readvariableopCsavev2_adam_dueling_deep_q_network_dense_bias_m_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_1_kernel_m_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_1_bias_m_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_2_kernel_m_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_2_bias_m_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_3_kernel_m_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_3_bias_m_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_kernel_v_read_readvariableopCsavev2_adam_dueling_deep_q_network_dense_bias_v_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_1_kernel_v_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_1_bias_v_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_2_kernel_v_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_2_bias_v_read_readvariableopGsavev2_adam_dueling_deep_q_network_dense_3_kernel_v_read_readvariableopEsavev2_adam_dueling_deep_q_network_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ü
_input_shapesê
ç: :	::
::	::	;:;: : : : : : : :	::
::	::	;:;:	::
::	::	;:;: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	;: 

_output_shapes
:;:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	;: 

_output_shapes
:;:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	;: 

_output_shapes
:;: 

_output_shapes
: 
ý
½
&__inference_signature_wrapper_27769113
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	;
	unknown_6:;
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_277689472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Õ	
÷
E__inference_dense_3_layer_call_and_return_conditional_losses_27769011

inputs1
matmul_readvariableop_resource:	;-
biasadd_readvariableop_resource:;
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	;*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:;*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ;2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ;tensorflow/serving/predict:³y
	

dense1

dense2
V
A
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
V__call__
W_default_save_signature
*X&call_and_return_all_conditional_losses"
_tf_keras_modelõ{"name": "dueling_deep_q_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "DuelingDeepQNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [128, 20]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "DuelingDeepQNetwork"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.003000000026077032, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
È

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 20]}}
Î

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"©
_tf_keras_layer{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 128]}}
Ð

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"«
_tf_keras_layer{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 128]}}
Ó

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
___call__
*`&call_and_return_all_conditional_losses"®
_tf_keras_layer{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 59, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 128]}}
ã
#iter

$beta_1

%beta_2
	&decay
'learning_ratemFmGmHmImJmKmLmMvNvOvPvQvRvSvTvU"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
Ê
(non_trainable_variables
regularization_losses
)metrics
*layer_regularization_losses
	variables

+layers
trainable_variables
,layer_metrics
V__call__
W_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
6:4	2#dueling_deep_q_network/dense/kernel
0:.2!dueling_deep_q_network/dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
-non_trainable_variables
regularization_losses
.metrics
/layer_regularization_losses
	variables

0layers
trainable_variables
1layer_metrics
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
9:7
2%dueling_deep_q_network/dense_1/kernel
2:02#dueling_deep_q_network/dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
2non_trainable_variables
regularization_losses
3metrics
4layer_regularization_losses
	variables

5layers
trainable_variables
6layer_metrics
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
8:6	2%dueling_deep_q_network/dense_2/kernel
1:/2#dueling_deep_q_network/dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
7non_trainable_variables
regularization_losses
8metrics
9layer_regularization_losses
	variables

:layers
trainable_variables
;layer_metrics
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
8:6	;2%dueling_deep_q_network/dense_3/kernel
1:/;2#dueling_deep_q_network/dense_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
<non_trainable_variables
regularization_losses
=metrics
>layer_regularization_losses
 	variables

?layers
!trainable_variables
@layer_metrics
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ô
	Btotal
	Ccount
D	variables
E	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 16}
:  (2total
:  (2count
.
B0
C1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
;:9	2*Adam/dueling_deep_q_network/dense/kernel/m
5:32(Adam/dueling_deep_q_network/dense/bias/m
>:<
2,Adam/dueling_deep_q_network/dense_1/kernel/m
7:52*Adam/dueling_deep_q_network/dense_1/bias/m
=:;	2,Adam/dueling_deep_q_network/dense_2/kernel/m
6:42*Adam/dueling_deep_q_network/dense_2/bias/m
=:;	;2,Adam/dueling_deep_q_network/dense_3/kernel/m
6:4;2*Adam/dueling_deep_q_network/dense_3/bias/m
;:9	2*Adam/dueling_deep_q_network/dense/kernel/v
5:32(Adam/dueling_deep_q_network/dense/bias/v
>:<
2,Adam/dueling_deep_q_network/dense_1/kernel/v
7:52*Adam/dueling_deep_q_network/dense_1/bias/v
=:;	2,Adam/dueling_deep_q_network/dense_2/kernel/v
6:42*Adam/dueling_deep_q_network/dense_2/bias/v
=:;	;2,Adam/dueling_deep_q_network/dense_3/kernel/v
6:4;2*Adam/dueling_deep_q_network/dense_3/bias/v
2
9__inference_dueling_deep_q_network_layer_call_fn_27769044Å
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
á2Þ
#__inference__wrapped_model_27768947¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
¡2
T__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_27769022Å
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
Ò2Ï
(__inference_dense_layer_call_fn_27769122¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_layer_call_and_return_conditional_losses_27769133¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_1_layer_call_fn_27769142¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_1_layer_call_and_return_conditional_losses_27769153¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_2_layer_call_fn_27769162¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_2_layer_call_and_return_conditional_losses_27769172¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_3_layer_call_fn_27769181¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_3_layer_call_and_return_conditional_losses_27769191¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
&__inference_signature_wrapper_27769113input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#__inference__wrapped_model_27768947q0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ;§
E__inference_dense_1_layer_call_and_return_conditional_losses_27769153^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_1_layer_call_fn_27769142Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_dense_2_layer_call_and_return_conditional_losses_27769172]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_dense_2_layer_call_fn_27769162P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_dense_3_layer_call_and_return_conditional_losses_27769191]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ;
 ~
*__inference_dense_3_layer_call_fn_27769181P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ;¤
C__inference_dense_layer_call_and_return_conditional_losses_27769133]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_layer_call_fn_27769122P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ»
T__inference_dueling_deep_q_network_layer_call_and_return_conditional_losses_27769022c0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ;
 
9__inference_dueling_deep_q_network_layer_call_fn_27769044V0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ;¦
&__inference_signature_wrapper_27769113|;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ;