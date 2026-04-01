import json, pathlib
nb = json.loads(pathlib.Path('notebooks/04_sensitivity_stats.ipynb').read_text('utf-8'))

c3 = ''.join(nb['cells'][2]['source'])
for key in ['bl_abs_W_event_ratio', 'bl_RW_mean_all', 'bl_RW_mean_event', 'bl_RW_mean_nonevent',
            'abs_width_event_ratio', 'relative_width_mean_event', 'relative_width_mean_nonevent', 'relative_width_mean']:
    print(f'  cell 3 has {key}: {key in c3}')

print()
c20 = ''.join(nb['cells'][19]['source'])
for label in ['Absolute width event ratio', 'Mean relative width (all days)', 'Mean relative width (event days)']:
    print(f'  cell 20 has "{label}": {label in c20}')

print()
c22 = ''.join(nb['cells'][21]['source'])
for label in ['Abs. width', 'Event ratio', 'mean, all']:
    print(f'  cell 22 has "{label}": {label in c22}')
