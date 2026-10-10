## @file scripts/integrate_training.py
# @brief integrate training source/document maintenance tool.
# @ingroup maintenance
# @details Textual maintenance only; does not itself validate numerical correctness.

"""Idempotently integrate trainable Transformer tests and preserve original CMake."""
from pathlib import Path
import runpy
runpy.run_path('scripts/harden_training.py')
p=Path('CMakeLists.txt');s=p.read_text()
if 'NAME transformer_training' not in s:
 s+='''
# Actual dataset trainer, available independently of BUILD_TESTING.
add_executable(cfwx_train test/train_transformer.c ${SOURCES_EXPERIMENTS})
target_link_libraries(cfwx_train PRIVATE cfwx_core cfwx_logger)
if(BUILD_TESTING)
    add_executable(cfwx_training_test_runner test/training_test.c)
    target_link_libraries(cfwx_training_test_runner PRIVATE cfwx_core)
    add_test(NAME transformer_gradients COMMAND cfwx_training_test_runner)
    add_test(NAME transformer_training COMMAND cfwx_train)
endif()
if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target cfwx_train cfwx_training_test_runner)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
        endif()
    endforeach()
endif()
'''
if 'NAME transformer_training_extra' not in s:
 s+='''
if(BUILD_TESTING)
    add_executable(cfwx_training_extra_test_runner test/training_extra_test.c)
    target_link_libraries(cfwx_training_extra_test_runner PRIVATE cfwx_core)
    add_test(NAME transformer_training_extra COMMAND cfwx_training_extra_test_runner)
    if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(cfwx_training_extra_test_runner PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        set_property(TARGET cfwx_training_extra_test_runner APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
    endif()
endif()
'''
p.write_text(s)
p=Path('include/models/transformer.h');s=p.read_text().replace('This module is forward-only except activation, embedding and loss gradients.','Analytic composite backward APIs are declared in models/training.h.').replace('cross-attention or a trainable autodiff graph.','cross-attention. Training APIs are in models/training.h.');p.write_text(s)
p=Path('TRANSFORMER_BLOCKS.md');s=p.read_text()
if '## Analytic training update' not in s:
 s+='''
## Analytic training update
The former forward-only limitation is superseded by `models/training.h`.
Complete analytic gradients cover attention (including additive bias), projected multi-head
attention, LayerNorm/RMSNorm, RoPE, gated/ungated FFNs and residual Transformer blocks.
`cfwx_language_model_loss` differentiates embeddings, stacked blocks, final normalization
and the vocabulary head. See TRAINING.md and ARCHITECTURE.md. Earlier forward-only notes
refer to the original forward API surface, not the current repository capabilities.
'''
p.write_text(s)
p=Path('README.md');s=p.read_text()
if '## End-to-end Transformer training' not in s:
 s+='''
## End-to-end Transformer training
The repository now includes complete analytic attention, normalization and composite-block
backward passes, a stacked causal language model, global gradient clipping and AdamW.
Run `cfwx_train` for the built-in overfit regression or provide a small byte dataset.
See [TRAINING.md](TRAINING.md) for commands, reduction/tokenization semantics and limitations,
and [ARCHITECTURE.md](ARCHITECTURE.md) for the repository/dependency overview.
The previous forward-only limitation is superseded by `models/training.h`.
'''
p.write_text(s)
