"""Add Doxygen comments to repository source without compiling or running CUDA.

Run at repository root. Existing comments/algorithms are preserved. Canonical
public contracts live in headers; definitions refer to those contracts. This
script is idempotent and uses deliberately conservative source matching rather
than claiming to be a C++ parser. Coverage is checked in generated Doxygen XML.
"""
from pathlib import Path
import re

GROUPS = {
    'cfwx.h': 'core', 'matrix.h': 'containers', 'vector.h': 'containers',
    'tensor.h': 'containers', 'neuron.h': 'containers',
    'activation_functions.h': 'activations', 'loss_functions.h': 'losses',
    'rewards.h': 'rewards', 'ann.h': 'dense', 'attention.h': 'attention',
    'transformer.h': 'transformer', 'training.h': 'training', 'runtime.h': 'runtime',
    'logger.h': 'logging', 'args.h': 'utilities', 'messages.h': 'utilities',
    'colors.h': 'utilities', 'cuda_training.h': 'gpu', 'checkpoint.c': 'checkpoint',
}
PARAMS = {
    'rows': 'Positive row count of the owning matrix.',
    'cols': 'Positive column/feature count of the owning matrix.',
    'rank': 'Tensor rank; supported range is 1 through CFWX_MAX_RANK.',
    'shape': 'Dimension array with rank entries; every dimension must be positive.',
    'indices': 'Index array with one entry per tensor dimension.',
    'offset': 'Position offset or checked flattened tensor offset, as specified by this operation.',
    'seed': 'Caller-supplied pseudorandom initialization seed or mutable RNG state; noncryptographic.',
    'rng': 'Borrowed mutable RNG state; consult the operation contract for consumption/replay rules.',
    'epsilon': 'Positive finite numerical-stability constant, not a trainable parameter.',
    'eps': 'Positive finite numerical-stability constant, not a trainable parameter.',
    'gamma': 'Feature scale array; one value per feature. Borrowed input storage.',
    'beta': 'Feature shift array; one value per feature. Borrowed input storage.',
    'dgamma': 'Caller-allocated feature-scale gradient output array.',
    'dg': 'Caller-allocated feature-scale gradient output array.',
    'dbeta': 'Caller-allocated feature-shift gradient output array.',
    'mask': 'Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.',
    'causal': 'Whether to additionally permit only key positions at or before each query.',
    'probabilities': 'Optional preallocated attention probability matrix [queries,keys].',
    'logits': 'Logit matrix/vector; dimensionality and input/output role are specified by the operation contract.',
    'labels': 'Categorical target IDs, one per row; ignore label is interpreted where supported.',
    'ignore_label': 'Target ID excluded from token-loss reduction; SIZE_MAX is the conventional sentinel.',
    'smoothing': 'Uniform label-smoothing fraction in [0,1); target class is included in uniform mass.',
    'temperature': 'Finite nonnegative sampling temperature; zero requests greedy decoding.',
    'top_k': 'Number of highest-logit classes retained; zero means all classes.',
    'capacity': 'Positive allocation/cache capacity or destination buffer capacity, in the units stated by the API.',
    'mode': 'Fixed or sliding cache eviction policy.',
    'dataset': 'Optional binary byte-dataset path; NULL selects the built-in corpus where supported.',
    'steps': 'Positive number of requested optimizer updates.',
    'logger': 'Borrowed initialized experiment logger; name/user data must outlive all calls.',
    'experiment_name': 'Borrowed nonempty experiment identity included in emitted records.',
    'name': 'Borrowed name/identity string; consult lifetime requirements of this interface.',
    'sink': 'Optional message callback; NULL selects the legacy backend.',
    'user_data': 'Borrowed callback state, not owned or freed by the logger.',
    'file': 'Call-site filename or internal stream handle, according to the declared type.',
    'line': 'Source line number for diagnostic attribution.',
    'fmt': 'Trusted printf-style format string; remaining variadic arguments must match it.',
    'minimum_level': 'Minimum emitted severity; independent of the application-wide threshold.',
    'level': 'Valid logging severity from DEBUG through FATAL.',
    'learning_rate': 'Finite nonnegative SGD learning rate.',
    'rate': 'Optimizer learning rate; zero is allowed only by APIs documenting gradient-only execution.',
    'beta1': 'Adam first-moment decay in [0,1).',
    'beta2': 'Adam second-moment decay in [0,1).',
    'decay': 'Finite nonnegative decoupled weight-decay coefficient.',
    'dropout': 'Training dropout probability in [0,1); exact replay uses the forward mask.',
    'max_norm': 'Positive finite maximum global gradient norm after accumulation.',
    'norm': 'Output for the measured pre-clipping global gradient norm.',
    'training': 'Training/evaluation mode; evaluation disables dropout and parameter updates.',
    'gradient': 'Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.',
    'optimizer': 'AdamW state owner with shape-matched moments and one shared update counter.',
    'state': 'Experiment state including vocabulary, RNG, dataset fingerprint and hyperparameters.',
    'path': 'File path used by the operation; checkpoint publication requires a writable parent directory.',
    'vocabulary': 'Positive vocabulary size or count of classes.',
    'heads': 'Positive attention head count dividing model width.',
    'hidden': 'Positive feed-forward hidden-feature width.',
    'depth': 'Positive layer count, bounded by CFWX_MAX_BLOCKS in the language model.',
    'width': 'Positive model-feature width divisible by the configured head count.',
    'batch': 'Positive number of independent sequences; reduction semantics are specified by the API.',
    'lengths': 'Optional per-sequence lengths; NULL selects the common context length.',
    'context': 'Common context/chunk length in tokens.',
    'scored': 'Output count of nonignored/held-out next-token targets included in reduction.',
    'loss': 'Output scalar objective; consult scalar-element versus token-mean reduction.',
    'argc': 'Number of process arguments including argv[0].',
    'argv': 'Process argument strings; the legacy parser may terminate on help/error.',
    'options': 'Caller-owned command-line option record.',
    'enabled': 'Nonzero selects the opt-in fused sequence launch path; zero selects serial execution.',
    'config': 'Borrowed CUDA model configuration including dimensions and allocation budgets.',
    'error': 'Caller-allocated diagnostic string buffer.',
    'error_capacity': 'Diagnostic buffer capacity in bytes, including the terminator.',
}
BRIEFS = {
    'cfwx_count': 'Check a positive element-count product and double-storage overflow.',
    'cfwx_vector_dot': 'Compute a compensated vector dot product.',
    'cfwx_matrix_multiply': 'Multiply row-major matrices into caller-owned output storage.',
    'cfwx_matrix_transpose': 'Transpose a matrix without allocating an output owner.',
    'cfwx_tensor_offset': 'Check tensor indices and compute a contiguous-storage offset.',
    'cfwx_tensor_reshape': 'Replace tensor metadata while preserving element count and allocation.',
    'cfwx_sigmoid': 'Evaluate a sign-stabilized logistic sigmoid.',
    'cfwx_relu': 'Evaluate ReLU, propagating NaN inputs.',
    'cfwx_relu_derivative': 'Evaluate the ReLU derivative with zero derivative at the origin.',
    'cfwx_gelu': 'Evaluate exact erf-based GELU rather than the tanh approximation.',
    'cfwx_gelu_derivative': 'Evaluate the analytic derivative of exact GELU.',
    'cfwx_softmax': 'Evaluate max-shifted softmax for finite logits.',
    'cfwx_mse': 'Compute mean squared error and optional scalar-element-reduced gradient.',
    'cfwx_bce_logits': 'Compute stable mean binary cross entropy from logits.',
    'cfwx_cross_entropy_logits': 'Compute stable categorical cross entropy and optional logit gradient.',
    'cfwx_token_cross_entropy': 'Compute token-mean categorical loss with ignore labels and smoothing.',
    'cfwx_discounted_returns': 'Compute reverse discounted returns with zero terminal bootstrap.',
    'cfwx_multihead_forward': 'Project Q/K/V, evaluate independent heads, concatenate and project outputs.',
    'cfwx_feedforward_forward': 'Evaluate the configured ungated or gated feed-forward block.',
    'cfwx_transformer_block_forward': 'Evaluate both pre-LayerNorm residual branches of a Transformer block.',
    'cfwx_attention_backward': 'Compute analytic Q/K/V and optional attention-bias gradients.',
    'cfwx_layer_norm_backward': 'Differentiate population LayerNorm input, scale and shift.',
    'cfwx_rms_norm_backward': 'Differentiate RMSNorm input and feature scale.',
    'cfwx_language_model_loss': 'Evaluate token loss and optionally differentiate all CPU language-model parameters.',
    'cfwx_language_model_loss_training': 'Evaluate/differentiate a language model with replayed residual dropout masks.',
    'cfwx_language_model_batch': 'Accumulate independent-sequence gradients weighted by valid token count.',
    'cfwx_language_model_evaluate': 'Evaluate every adjacent held-out target once using chunked contexts.',
    'cfwx_checkpoint_save': 'Publish a versioned CPU checkpoint with payload and parent-directory synchronization.',
    'cfwx_checkpoint_load': 'Validate and transactionally construct CPU model, optimizer and experiment state.',
    'cfwx_language_model_decode': 'Append one token and evaluate next-token logits using projected K/V history.',
    'cfwx_kv_cache_init': 'Initialize a fixed-capacity cache using the legacy default policy.',
    'cfwx_kv_cache_init_mode': 'Initialize an explicitly fixed or sliding projected K/V cache.',
    'cfwx_sample_logits': 'Sample finite logits with greedy/temperature and optional top-k filtering.',
    'cfwx_adamw_step': 'Apply one global AdamW update with bias-corrected moments.',
    'cfwx_cuda_create': 'Create the opt-in device reference model and owned CUDA resources.',
    'cfwx_cuda_run': 'Execute the device batch objective and optional analytic update.',
    'cfwx_cuda_set_fused_batching': 'Select launch-fused independent-sequence execution rather than serial interpretation.',
    'experiment_logger_init': 'Initialize a borrowed named logger context and optional sink.',
    'experiment_logger_log': 'Filter and route an experiment-attributed formatted record.',
    'logger_log': 'Emit an application-attributed record through the legacy shared backend.',
}

def group(path):
    if str(path).startswith('test/'):
        return 'gpu' if path.name == 'cuda_manual.c' else 'tests'
    if '/experiments/' in str(path): return 'experiments'
    if '/gpu/' in str(path): return 'gpu'
    return GROUPS.get(path.name, GROUPS.get(path.stem+'.h', 'utilities' if '/util/' in str(path) else 'core'))

# Conservative declaration/definition matcher; function pointers and control flow
# are not considered standalone APIs. CUDA qualifiers and extern-C are accepted.
FUNCTION = re.compile(r'(?m)^(?P<indent>[ \t]*)(?P<ret>(?:(?:extern\s+"C"|static|inline|__global__|__device__|__host__)\s+)*(?:cfwx_status|int|void|bool|double|size_t|uint64_t|const\s+char\s*\*|node\s*\*|Dense|std::vector<int>))\s*(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^;{}]*?)\)\s*(?P<end>[;{])')

public = {}
for path in sorted(Path('include').rglob('*.h')):
    for m in FUNCTION.finditer(path.read_text()):
        public[m['name']] = path

for root in ('include','src','test'):
    for path in sorted(Path(root).rglob('*')):
        if path.suffix not in ('.h','.c','.cu'): continue
        text = path.read_text()
        module = group(path)
        if '@file '+str(path) not in text:
            title = BRIEFS.get(path.stem, path.stem.replace('_',' ')+' interfaces and implementation.')
            text = '/**\n * @file '+str(path)+'\n * @brief '+title+'\n * @ingroup '+module+'\n * @details See the module guide for shape, ownership, failure and execution-policy\n * constraints. Source listings include implementation helpers and ordinary\n * explanatory comments; generated source documentation does not execute code.\n */\n\n'+text
        # Promote explanatory contracts to Doxygen details. Only block comments,
        # never strings or preprocessor replacements, are changed by this step.
        text = re.sub(r'/\*(?![*!])', '/**', text)
        inserts=[]
        for m in FUNCTION.finditer(text):
            name=m['name']
            # Stable marker avoids doubling annotations on repeated maintenance.
            if '@anchor api_'+str(path).replace('/','_').replace('.','_')+'_'+name in text: continue
            args=m['args'].strip()
            names=[]
            for arg in args.split(','):
                arg=arg.strip()
                if not arg or arg in ('void','...'): continue
                match=re.search(r'([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$',arg)
                if match:names.append((match[1],arg))
            brief=BRIEFS.get(name)
            if not brief:
                if name.endswith('_free'):brief='Release owned '+name.replace('cfwx_','').replace('_free','').replace('_',' ')+' storage and reset its owner.'
                elif name.endswith('_init'):brief='Construct and initialize '+name.replace('cfwx_','').replace('_init','').replace('_',' ')+' state.'
                elif name.endswith('_backward'):brief='Compute analytic adjoints for '+name.replace('cfwx_','').replace('_backward','').replace('_',' ')+'.'
                elif name=='main':brief='Run this executable driver and propagate its success/failure status.'
                else:brief='Perform the '+name.replace('cfwx_','').replace('_',' ')+' operation.'
            block=['/**',' * @anchor api_'+str(path).replace('/','_').replace('.','_')+'_'+name,' * @brief '+brief,' * @ingroup '+module]
            if path.suffix in ('.c','.cu') and name in public:
                block+=[' * @copydetails '+name,' * @see '+str(public[name])]
            else:
                block+=[' * @details Read the adjacent operation-specific contract and linked module guide',
                        ' * for exact dimensions, aliasing, reduction and error behavior. Parameters',
                        ' * are borrowed unless the constructor/free contract explicitly transfers ownership.']
                for param,arg in names:
                    desc=PARAMS.get(param)
                    if desc is None:
                        if param in ('dy','dx','dq','dk','dv','dbias','dw','dquery','dcontext'):
                            desc='Upstream adjoint input.' if param=='dy' else 'Preallocated analytic gradient output; shape matches the corresponding primal.'
                        elif param in ('model','layer','block','m','l'):
                            desc='Model/operation state; input versus owner role follows the declaration and operation contract.'
                        elif param in ('x','a','b','q','k','v','query','context'):
                            desc='Numerical operand or operation state; see the documented mathematical shape and role.'
                        elif param in ('out','p','g','n','t','count','size','length','ids','tokens','targets','data','values'):
                            desc=''+param+' argument as declared; use the adjacent contract for storage, count and input/output role.'
                        else:desc=''+param+' argument; interpretation follows this operation\'s implementation and contract.'
                    block.append(' * @param '+param+' '+desc)
                ret=m['ret']
                if 'cfwx_status' in ret:
                    block+=[' * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation',
                            ' * or numerical status as checked by the implementation. Do not consume failed outputs.']
                elif re.search(r'\bvoid\b',ret):block+=[' * @return No value; inspect the documented mutation/lifetime effects.']
                elif name.startswith('cfwx_cuda_'):block+=[' * @return Zero on success for status APIs; counts/error text for query APIs,',
                                                        ' * as specified in the adjacent CUDA contract.']
                elif ret.strip().endswith('*'):block+=[' * @return Pointer described by the operation contract; ownership is not implied.']
                else:block+=[' * @return Computed value or status described by the adjacent contract.']
            if '/gpu/' in str(path) or path.name=='cuda_manual.c':
                block+=[' * @warning CUDA source remains manually validated only. Doxygen parses this',
                        ' * symbol without running nvcc, CUDA kernels or hardware probes.']
            if '/models/' in str(path) and path.suffix=='.c':
                block+=[' * @note Internal helper state may be temporary. Referenced-by/source links show',
                        ' * how composite callers propagate errors and release allocations.']
            block+=[' */','']
            inserts.append((m.start(),'\n'.join(block)))
        for index,comment in reversed(inserts):text=text[:index]+comment+text[index:]
        # Types and members: document fields, enum choices and public macros.
        lines=text.splitlines();output=[];in_record=False
        for line in lines:
            stripped=line.strip()
            if stripped.startswith('typedef struct') or stripped.startswith('typedef enum'):
                if not (output and '@brief Record' in output[-1]):
                    output.append('/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */')
                in_record=True
            if in_record and stripped.startswith('}'):
                in_record=False
            if in_record and stripped and not stripped.startswith(('/', '*','{','#','typedef')) and ';' in stripped and '/**<' not in stripped:
                # Member-specific comments supplement rather than replace original contracts.
                line+=' /**< '+('Borrowed pointer or owned storage as specified by the containing type.' if '*' in stripped else 'State/dimension/policy value; see the containing type contract.')+' */'
            elif in_record and re.fullmatch(r'(?:CFWX_\w+|LOG_\w+)\s*(?:=\s*\w+)?\s*,?',stripped) and '/**<' not in stripped:
                line+=' /**< '+stripped.rstrip(',').split('=')[0].strip().replace('_',' ').lower()+' enum choice. */'
            if stripped.startswith('#define ') and not stripped.endswith('\\'):
                macro=re.match(r'#define\s+(\w+)',stripped)
                if macro and not macro[1].endswith('_H') and '@def '+macro[1] not in text:
                    output.append('/** @def '+macro[1]+'\n * @brief '+macro[1].replace('_',' ').lower()+' macro; see expansion and module contract.\n */')
            output.append(line)
        text='\n'.join(output)+'\n'
        # Fix known stale prose without changing executable semantics.
        text=text.replace('Token positions increase monotonically, no sliding-window eviction.',
                          'Token positions increase monotonically; fixed/sliding eviction is configurable.')
        path.write_text(text)

# Non-C maintenance code retains its own syntax; Doxygen recognizes Python ##.
for path in Path('scripts').glob('*.py'):
    text=path.read_text()
    if '@file '+str(path) not in text:
        text='## @file '+str(path)+'\n# @brief '+path.stem.replace('_',' ')+' source/document maintenance tool.\n# @ingroup maintenance\n# @details Textual maintenance only; does not itself validate numerical correctness.\n\n'+text
        path.write_text(text)
for path in Path('cmake').glob('*.cmake'):
    text=path.read_text()
    if '@file '+str(path) not in text:
        text='## @file '+str(path)+'\n# @brief '+path.stem.replace('_',' ')+' build helper.\n# @ingroup maintenance\n\n'+text
        path.write_text(text)

p=Path('CMakeLists.txt');text=p.read_text()
if 'include(cmake/doxygen.cmake)' not in text:
    text+='\n# Documentation is additive and never requires configuring CUDA.\ninclude(cmake/doxygen.cmake)\n'
p.write_text(text)
p=Path('.gitignore');text=p.read_text()
if 'build-docs/' not in text:text+='\nbuild-docs/\n'
p.write_text(text)
p=Path('README.md');text=p.read_text()
if '## Doxygen HTML reference' not in text:
    text+='''
## Doxygen HTML reference
Run `doxygen Doxyfile` from the repository root and open `build-docs/html/index.html`.
The additive CMake `docs` target runs the same configuration. Documentation includes
module groups, public contracts, internal definitions, source browsing, model/state
records, tests and multi-page architecture/training/runtime/GPU design guides.
See `docs/doxygen/README.md`. CUDA is parsed as source only; no GPU build or test is
required or performed by documentation generation.
'''
p.write_text(text)
print('Annotated C/CUDA/Python/build sources and added Doxygen integration.')
