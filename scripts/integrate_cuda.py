"""Static source maintenance only: NEVER configures, compiles or executes CUDA."""
from pathlib import Path
p=Path('src/gpu/cuda_training.cu');s=p.read_text()
if 'dropmask' not in s:
 s=s.replace('double *value=nullptr,*adjoint=nullptr;','double *value=nullptr,*adjoint=nullptr;\n    double *dropmask=nullptr;')
 s=s.replace('GELU, DROP, LOOKUP };','GELU, DROP, LOOKUP, MAXIMUM };')
 s=s.replace('bool training,uint64_t rng,int *status)','bool training,uint64_t rng,double *dropmask,int *status)')
 s=s.replace('Node x=nodes[i];double a=x.a>=0?v[x.a]:0.0,b=x.b>=0?v[x.b]:0.0,y=0.0;', 'Node x=nodes[i];\n        bool parents=x.op!=CONSTANT && x.op!=PARAMETER && x.op!=LOOKUP;\n        double a=parents && x.a>=0?v[x.a]:0.0,b=parents && x.b>=0?v[x.b]:0.0,y=0.0;')
 start=s.index('            case DROP:\n                y=')
 end=s.index('\n                break;',start)+len('\n                break;')
 s=s[:start]+'''            case MAXIMUM:y=fmax(a,b);break;
            case DROP:
                dropmask[i]=training && x.constant>0.0 ?
                    ((double)(next_random(rng)>>11)*0x1.0p-53<x.constant?0.0:1.0/(1.0-x.constant)):1.0;
                y=a*dropmask[i];break;'''+s[end:]
 s=s.replace('size_t parameter_count,bool training,int *status)', 'size_t parameter_count,bool training,const double *dropmask,int *status)')
 s=s.replace('double a=x.a>=0?v[x.a]:0.0,b=x.b>=0?v[x.b]:0.0;', 'bool parents=x.op!=CONSTANT && x.op!=PARAMETER && x.op!=LOOKUP;\n        double a=parents && x.a>=0?v[x.a]:0.0,b=parents && x.b>=0?v[x.b]:0.0;')
 start=s.index('            case DROP:\n                /*')
 end=s.index('\n            default:break;',start)
 s=s[:start]+'''            case DROP:g[x.a]+=u*dropmask[i];break;
            /* Detached max shift has zero adjoint. Shift contributions cancel
             * exactly in softmax/log-sum-exp, including maximum ties. */
            case MAXIMUM:break;'''+s[end:]
 s=s.replace('add(12,','add(MAXIMUM,')
 s=s.replace('cudaFree(m->adjoint);','cudaFree(m->adjoint);cudaFree(m->dropmask);')
 s=s.replace('sizeof(Node)+2*sizeof(double)','sizeof(Node)+3*sizeof(double)')
 s=s.replace('m->adjoint=nullptr;m->capacity=0;','m->adjoint=nullptr;m->dropmask=nullptr;m->capacity=0;')
 s=s.replace('cuda_check(cudaMalloc((void**)&m->adjoint,n*sizeof(double)));m->capacity=n;', 'cuda_check(cudaMalloc((void**)&m->adjoint,n*sizeof(double)));\n            cuda_check(cudaMalloc((void**)&m->dropmask,n*sizeof(double)));m->capacity=n;')
 s=s.replace('training!=0,m->rng,m->status);','training!=0,m->rng,m->dropmask,m->status);')
 s=s.replace('m->count,true,m->status);','m->count,true,m->dropmask,m->status);')
 # Advance stream by the number of RNG draws, rather than sliding by a single draw.
 s=s.replace('if(training){m->rng+=UINT64_C(0x9e3779b97f4a7c15);if(rate>0)m->step++;}', '''if(training){
            size_t draws=0;
            for(const Node &node:graph.nodes)if(node.op==DROP && node.constant>0.0)draws++;
            m->rng+=UINT64_C(0x9e3779b97f4a7c15)*(uint64_t)draws;
            if(rate>0)m->step++;
        }''')
 p.write_text(s)
p=Path('CMakeLists.txt');s=p.read_text()
# Global GNU-only flags must not leak to nvcc or MSVC.
s=s.replace('add_compile_options(-Wall -Wextra -pedantic -fstack-protector-all -Wstack-protector)', '''if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall> $<$<COMPILE_LANGUAGE:C>:-Wextra>
        $<$<COMPILE_LANGUAGE:C>:-pedantic> $<$<COMPILE_LANGUAGE:C>:-fstack-protector-all>
        $<$<COMPILE_LANGUAGE:C>:-Wstack-protector>)
endif()''')
if 'include(cmake/cuda_backend.cmake)' not in s:s+='\n# CUDA is explicitly opt-in; no CUDA configuration in ordinary CPU builds.\ninclude(cmake/cuda_backend.cmake)\n'
p.write_text(s)
# Source formatting only; no nvcc/toolkit lookup or execution.
import subprocess
subprocess.run(['clang-format','-i','src/gpu/cuda_training.cu','include/gpu/cuda_training.h','test/cuda_manual.c'],check=True)
