P
đ
Ă

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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ł
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
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
Á
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
executor_typestring ¨
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
StringLower	
input

output"
encodingstring "serve*2.10.12v2.10.0-76-gfdfc646704c86
y
serving_default_inputsPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
{
serving_default_inputs_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
ó
PartitionedCallPartitionedCallserving_default_inputsserving_default_inputs_1*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_signature_wrapper_121

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ę
valueŔB˝ Bś

created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures* 
* 
* 
* 
* 
* 
* 

serving_default* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
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
GPU 2J 8 *%
f R
__inference__traced_save_148

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
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
GPU 2J 8 *(
f#R!
__inference__traced_restore_158é)
Č	
{
!__inference_signature_wrapper_121

inputs
inputs_1	
identity

identity_1

identity_2	

identity_3	Š
PartitionedCallPartitionedCallinputsinputs_1*
Tin
2	*
Tout
2		*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *
fR
__inference_pruned_107`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b

Identity_2IdentityPartitionedCall:output:2*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b

Identity_3IdentityPartitionedCall:output:3*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1
Ä
p
__inference_pruned_107

inputs
inputs_1	
identity

identity_1

identity_2	

identity_3	Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙\
IdentityIdentityinputs_copy:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
StringLowerStringLowerinputs_copy:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙^

Identity_1IdentityStringLower:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_1_copyIdentityinputs_1*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identityinputs_1_copy:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_3Identityinputs_1_copy:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

i
__inference__traced_save_148
file_prefix
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B °
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
Ć
E
__inference__traced_restore_158
file_prefix

identity_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ł
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"ľ	J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultö
9
inputs/
serving_default_inputs:0˙˙˙˙˙˙˙˙˙
=
inputs_11
serving_default_inputs_1:0	˙˙˙˙˙˙˙˙˙4
sentence(
PartitionedCall:0˙˙˙˙˙˙˙˙˙7
sentence_xf(
PartitionedCall:1˙˙˙˙˙˙˙˙˙5
	sentiment(
PartitionedCall:2	˙˙˙˙˙˙˙˙˙8
sentiment_xf(
PartitionedCall:3	˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:Ý


created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.B,
__inference_pruned_107inputsinputs_1
,
serving_default"
signature_map
ĎBĚ
!__inference_signature_wrapper_121inputsinputs_1"
˛
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
annotationsŞ *
 ÷
__inference_pruned_107Ü˘
x˘u
sŞp
5
sentence)&
inputs/sentence˙˙˙˙˙˙˙˙˙
7
	sentiment*'
inputs/sentiment˙˙˙˙˙˙˙˙˙	
Ş "ÔŞĐ
.
sentence"
sentence˙˙˙˙˙˙˙˙˙
4
sentence_xf%"
sentence_xf˙˙˙˙˙˙˙˙˙
0
	sentiment# 
	sentiment˙˙˙˙˙˙˙˙˙	
6
sentiment_xf&#
sentiment_xf˙˙˙˙˙˙˙˙˙	č
!__inference_signature_wrapper_121Âi˘f
˘ 
_Ş\
*
inputs 
inputs˙˙˙˙˙˙˙˙˙
.
inputs_1"
inputs_1˙˙˙˙˙˙˙˙˙	"ÔŞĐ
.
sentence"
sentence˙˙˙˙˙˙˙˙˙
4
sentence_xf%"
sentence_xf˙˙˙˙˙˙˙˙˙
0
	sentiment# 
	sentiment˙˙˙˙˙˙˙˙˙	
6
sentiment_xf&#
sentiment_xf˙˙˙˙˙˙˙˙˙	