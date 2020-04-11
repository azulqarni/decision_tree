#ifndef ACCELERATOR_H_
#define ACCELERATOR_H_ 
#include "decision_tree_def.h"

#undef ZERO_COPY
#define EMULATION 
#define X 64
#define Y 4
#ifdef EMULATION
#define maxAtt X
#define maxClass Y
#else
#define maxAtt 52
#define maxClass 2
#endif

#ifdef ZERO_COPY
#pragma SDS data mem_attribute(Stable:CACHEABLE, Mtable:CACHEABLE, Classes:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(Stable:SEQUENTIAL, Mtable:SEQUENTIAL)
#pragma SDS data data_mover(Stable:AXIDMA_SIMPLE, Mtable:AXIDMA_SIMPLE)
#pragma SDS data copy(Stable[0:offset*length], Mtable[0:(width-1-offset)*length])
#pragma SDS data zero_copy(Classes[0:length])
#pragma SDS data sys_port(Stable:ACP, Mtable:ACP)
#else
#pragma SDS data mem_attribute(Stable:PHYSICAL_CONTIGUOUS, Mtable:PHYSICAL_CONTIGUOUS, Classes:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(Stable:SEQUENTIAL, Mtable:SEQUENTIAL, Classes:SEQUENTIAL)
#pragma SDS data data_mover(Stable:AXIDMA_SIMPLE, Mtable:AXIDMA_SIMPLE, Classes:AXIDMA_SIMPLE)
#pragma SDS data sys_port(Stable:ACP, Mtable:ACP, Classes:ACP)
#pragma SDS data copy(Stable[0:offset*length], Mtable[0:(width-1-offset)*length], Classes[0:length])
#endif
data_t evalDiscreteAtt_hw(data_t *Stable, data_t *Mtable, data_t *Classes,
                          int length, int width, int offset);
#endif
