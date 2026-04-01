import json
from pathlib import Path

NB_PATH = Path(r'c:/Users/Usuario/OneDrive - UNIVERSIDAD DE HUELVA/Granada/TrabajoFM/scripts/Python_Pipeline_SWAT_Pascal/swat_pipeline/trabajoFM/notebooks/04_sensitivity_stats.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_src(i): return ''.join(nb['cells'][i]['source'])
def set_src(i, t): nb['cells'][i]['source'] = t.splitlines(keepends=True)

# --- PATCH 1: helper cell (index 7) ---
src7 = get_src(7)
if 'annotation_formatter' not in src7:
    OLD_SIG = '    tight_layout_rect=(0.0, 0.02, 0.98, 0.98),\n\n\n\n):'
    NEW_SIG = '    tight_layout_rect=(0.0, 0.02, 0.98, 0.98),\n\n    annotation_formatter=None,   # optional callable: (value, stat, scenario, variable, default_text) -> str | None\n\n\n):'
    if OLD_SIG in src7:
        src7 = src7.replace(OLD_SIG, NEW_SIG, 1); print('P1a OK')
    else:
        idx = src7.find('tight_layout_rect=(0.0, 0.02, 0.98, 0.98),')
        print('P1a ERR:', repr(src7[idx:idx+60]))
    OLD_L = '                    label_txt = _format_value(y, stat)\n\n                subgroup_entries.append({'
    NEW_L = '                    label_txt = _format_value(y, stat)\n\n                if annotation_formatter is not None:\n                    _af_result = annotation_formatter(y, stat, scenario, var, label_txt)\n                    if _af_result is not None:\n                        label_txt = str(_af_result)\n\n                subgroup_entries.append({'
    if OLD_L in src7:
        src7 = src7.replace(OLD_L, NEW_L, 1); print('P1b OK')
    else:
        idx = src7.find('label_txt = _format_value(y, stat)')
        print('P1b ERR:', repr(src7[idx:idx+80]))
    set_src(7, src7)

# --- PATCH 2: chart 2 cell (index 21) ---
src22 = get_src(21)
if '_ch2_annotation_formatter' not in src22:
    i0 = src22.find('    if _ch2_compare_mode == \"conc\":')
    i1 = src22.find('        ch_2_y_label = \"shift vs. BASE (kg/day)\"', i0)
    if i0 == -1 or i1 == -1:
        print('P2a ERR: block not found i0=%d i1=%d' % (i0, i1))
    else:
        end = i1 + len('        ch_2_y_label = \"shift vs. BASE (kg/day)\"')
        NEW_BLOCK = '    if _ch2_compare_mode == \"conc\":\n\n        ch_2_chart_descriptor = \"Shift sensitivity compared to BASE scenario (concentrations)\"\n        ch_2_y_label = \"shift vs. BASE (mg/L)\"\n\n        def _ch2_annotation_formatter(value, stat, scenario, variable, default_text):\n            if variable == \"TP\":\n                return f\"{value:+.1f}\"\n            return default_text\n\n    else:\n\n        ch_2_chart_descriptor = \"Shift sensitivity compared to BASE scenario (total loads)\"\n        ch_2_y_label = \"shift vs. BASE (kg/day)\"\n        _ch2_annotation_formatter = None'
        src22 = src22[:i0] + NEW_BLOCK + src22[end:]
        print('P2a OK')
    idx_sub = src22.find('        subtitle_y=-0.0075,')
    if idx_sub == -1:
        print('P2b ERR: subtitle_y not found')
    else:
        idx_close = src22.find('\n    )\n', idx_sub)
        if idx_close == -1:
            print('P2b ERR: close not found:', repr(src22[idx_sub:idx_sub+200]))
        else:
            src22 = src22[:idx_close] + '\n\n        annotation_formatter=_ch2_annotation_formatter,' + src22[idx_close:]
            print('P2b OK')
    set_src(21, src22)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
s8  = ''.join(nb2['cells'][7]['source'])
s22 = ''.join(nb2['cells'][21]['source'])
print('VERIFY c8 param:', 'annotation_formatter=None' in s8)
print('VERIFY c8 hook:', '_af_result = annotation_formatter' in s8)
print('VERIFY c22 def:', 'def _ch2_annotation_formatter' in s22)
print('VERIFY c22 kwarg:', 'annotation_formatter=_ch2_annotation_formatter,' in s22)
