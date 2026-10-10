"""Register analytic gradient tests and real training runner, preserving existing targets."""
from pathlib import Path
p=Path('CMakeLists.txt');s=p.read_text()
if 'NAME transformer_training' not in s:
 s+='''
# Standalone dataset training executable, also available with BUILD_TESTING=OFF.
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
 p.write_text(s)
# Earlier capability notes are superseded by analytic training APIs.
p=Path('include/models/transformer.h');s=p.read_text().replace('This module is forward-only except activation, embedding and loss gradients.','Analytic composite backward APIs are declared in models/training.h.').replace('cross-attention or a trainable autodiff graph.','cross-attention. Training APIs are in models/training.h.');p.write_text(s)
p=Path('TRANSFORMER_BLOCKS.md');s=p.read_text();s+='''\n## Analytic training update\nThe former forward-only limitation is superseded by `models/training.h`. It provides complete analytic gradients for attention (including additive bias), projected multi-head attention, LayerNorm/RMSNorm, RoPE, gated/ungated FFNs and residual Transformer blocks. `cfwx_language_model_loss` differentiates embeddings, stacked blocks, final normalization and the vocabulary head. See TRAINING.md for dataset training and ARCHITECTURE.md for repository structure. Earlier forward-only descriptions apply only to the original forward API surface.\n''';p.write_text(s)
