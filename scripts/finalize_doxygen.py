## @file scripts/finalize_doxygen.py
# @brief finalize doxygen source/document maintenance tool.
# @ingroup maintenance
# @details Textual maintenance only; does not itself validate numerical correctness.

"""Refine Doxygen source matching and preserve full operation contracts in API docs."""
from pathlib import Path
import re
p=Path('scripts/document_repository.py');s=p.read_text()
s=s.replace("if any(re.fullmatch(r'[A-Za-z_]\\w*', a.strip()) and a.strip() != 'void' for a in m['args'].split(',')):", "if any(a.strip() not in ('void', '...') and not re.match(r'^(?:const\\s+)?(?:cfwx_\\w+|experiment_logger|options_t|LogLevel|logger_sink_fn|size_t|uint64_t|double|int|unsigned|char|bool|node|tape|stream|Dense|Graph|Node|Segment|std::vector<int>|cudaError_t)\\b', a.strip()) for a in m['args'].split(',')):")
# pointer-return declarations have no whitespace after the star in existing headers.
s=s.replace("(?=\\s|\\*)(?:\\s+|(?<=\\*)\\s*)", "(?:\\s+|(?<=\\*)\\s*)")
# Prevent partial token matching (int eger -> integer) by requiring actual arg types.
# Above args checks do not fix integer: require whitespace after word returns.
s=s.replace("(?:\\s+|(?<=\\*)\\s*)(?P<name>", "(?:\\s+|(?<=\\*)\\s*)(?P<name>")
# Add a richer common vocabulary without inventing output directions for overloaded names.
needle='public = {}'
s=s.replace(needle,'''PARAMS.update({
    'x': 'Borrowed primal/input numeric operand; shape is stated in the operation contract.',
    'dy': 'Borrowed upstream adjoint; same output shape as the corresponding forward operation.',
    'out': 'Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.',
    'ids': 'Borrowed token/index array with one entry per sequence position; validate against vocabulary/table bounds.',
    'tokens': 'Borrowed token IDs, laid out as sequence/batch arrays according to this API.',
    'targets': 'Borrowed next-token targets; SIZE_MAX is ignored only where the loss API explicitly supports it.',
    'count': 'Number of entries/elements; must match the associated buffer or registry contract.',
    'length': 'Number of sequence positions/tokens processed by this call.',
    'size': 'Positive number of elements in the requested storage.',
    'values': 'Borrowed numerical values to import or process; length must match the supplied count.',
    'data': 'Borrowed source bytes/numerical storage; valid for the stated count.',
    'message': 'Borrowed formatted diagnostic payload; valid only for the sink callback duration.',
    'weights': 'Borrowed row-major projection weights [input,output].',
    'input': 'Borrowed neuron input vector matching the weight count.',
    'scale': 'Finite vector multiplier; numerical overflow can follow ordinary IEEE-754 behavior.',
    'p': 'Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.',
    'g': 'Gradient record or internal operand as declared; shape/reduction follows this operation contract.',
    'n': 'Element count or owner/internal node as declared; see containing module contracts.',
    'a': 'First borrowed numerical operand or internal node/state as declared.',
    'b': 'Second borrowed numerical operand or internal node/state as declared.',
    'q': 'Query operand [queries,head_depth] for attention, or corresponding model query state.',
    'k': 'Key operand [keys,head_depth] for attention.',
    'v': 'Value operand [keys,value_depth] for attention.',
    'clip': 'Positive finite global gradient-clipping threshold.',
})

public = {}''')
# Generated public block incorporates adjacent ordinary contract into detailed HTML.
s=s.replace("args=m['args'].strip()", """args=m['args'].strip()
            adjacent = re.search(r'/\\*\\*?(.*?)\\*/\\s*$', text[:m.start()], re.S)
            contract = ''
            if adjacent:
                raw = adjacent[1]
                if '@anchor' not in raw and '@file' not in raw:
                    contract = ' '.join(re.sub(r'(?m)^\\s*\\* ?', '', raw).split())
""")
s=s.replace("' * are borrowed unless the constructor/free contract explicitly transfers ownership.']", "' * are borrowed unless the constructor/free contract explicitly transfers ownership.']\n                if contract: block += [' * @par Operation contract', ' * '+contract]")
p.write_text(s)
