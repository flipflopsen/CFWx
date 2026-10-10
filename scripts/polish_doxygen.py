## @file scripts/polish_doxygen.py
# @brief polish doxygen source/document maintenance tool.
# @ingroup maintenance
# @details Textual maintenance only; does not itself validate numerical correctness.

"""Source-only annotation quality improvements; does not execute CUDA or models."""
from pathlib import Path
p=Path('scripts/document_repository.py');s=p.read_text()
s=s.replace('(?:cfwx_\\w+|experiment_logger|options_t|LogLevel|logger_sink_fn|size_t|uint64_t|double|int|unsigned|char|bool|node|tape|stream|Dense|Graph|Node|Segment|std::vector<int>|cudaError_t)', '(?:cfwx_\\w+|experiment_logger|options_t|LogLevel|logger_sink_fn|size_t|uint64_t|double|int|unsigned|char|bool|void|node|tape|stream|Dense|Graph|Node|Segment|std::vector<int>|cudaError_t)')
s=s.replace("adjacent = re.search(r'/\\*\\*?(.*?)\\*/\\s*$', text[:m.start()], re.S)", "adjacent_matches = list(re.finditer(r'/\\*\\*?((?:(?!\\*/).)*?)\\*/', text[:m.start()], re.S))\n            adjacent = adjacent_matches[-1] if adjacent_matches and not text[adjacent_matches[-1].end():m.start()].strip() else None")
s=s.replace("desc=PARAMS.get(param)", '''desc=PARAMS.get(param)
                    # Names such as v/q/k are overloaded: inspect type and operation.
                    if 'cfwx_vector' in arg:
                        desc = 'Vector owner/operand with contiguous size-element storage; constructor/free versus input role follows this operation.'
                    if name.endswith('_init') and param in ('m','v','n','t','l','layer','block','model','cache','optimizer','o'):
                        desc = 'Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.'
                    if name.endswith('_free'):
                        desc = 'Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.'
                    if param == 'out' and name == 'cfwx_vector_dot':
                        desc = 'Scalar dot-product output pointer; written only after successful finite-result validation.'
                    if param == 'out' and name.startswith('cfwx_cuda_get_'):
                        desc = 'Host destination double array; count must equal the flattened CUDA parameter count.'
                    if param == 'out' and name == 'cfwx_cuda_create':
                        desc = 'Output pointer to a newly created opaque device model; initialized to NULL on entry.'
                    if param == 'n' and 'size_t' in arg:
                        desc = 'Positive scalar element count; each associated vector/array must contain at least n entries.'
                    if param in ('q','k','v') and 'cfwx_matrix' not in arg:
                        desc = 'Operation-specific owner or scalar as declared; not necessarily an attention operand.'
                    if param == 'logits' and name.endswith(('forward','decode')):
                        desc = 'Preallocated logit output matrix: [length,vocabulary] for full forward or [1,vocabulary] for decode.'
                    if param == 'token' and name == 'cfwx_language_model_decode':
                        desc = 'Input token ID below model vocabulary; appended to the cache on success.'
                    if param == 'token' and name == 'cfwx_sample_logits':
                        desc = 'Output sampled categorical token ID.'
''')
s=s.replace("elif name.startswith('cfwx_cuda_'):block+=[' * @return Zero on success for status APIs; counts/error text for query APIs,',\n                                                        ' * as specified in the adjacent CUDA contract.']", "elif name == 'cfwx_cuda_error':block+=[' * @return Borrowed last-error text; valid until the next model API call or destruction.']\n                elif name == 'cfwx_cuda_parameter_count':block+=[' * @return Flattened scalar parameter count; zero for a NULL model.']\n                elif name.startswith('cfwx_cuda_'):block+=[' * @return Zero on success, nonzero on invalid arguments, allocation, CUDA API or numerical failure.']")
# Include void* data members too.
s=s.replace('|logger_sink_fn|node|cudaStream_t)', '|logger_sink_fn|node|cudaStream_t|void)')
p.write_text(s)
