from pathlib import Path

dir_path = Path('C:/SWAT/RSWAT/cubillas/cubillas_set_219_ruben/cubillas_BASE_set-219/BASE recreated from arcswat default/TxtInOut_1')
chm_files = sorted(dir_path.glob('*.chm'))
print('CHM files found:', len(chm_files))
keys = [
    'Soil organic N [mg/kg]',
    'Soil NO3 [mg/kg]',
    'Soil organic P [mg/kg]',
    'Soil labile P [mg/kg]',
]
unique = {k: set() for k in keys}
missing = {k: 0 for k in keys}
for p in chm_files:
    text = p.read_text(encoding='utf-8', errors='ignore').splitlines()
    found = {k: False for k in keys}
    for line in text:
        if ':' not in line:
            continue
        label, after = line.split(':', 1)
        label = label.strip()
        if label in keys:
            found[label] = True
            toks = after.split()
            for tok in toks:
                try:
                    unique[label].add(float(tok))
                except ValueError:
                    pass
    for k, v in found.items():
        if not v:
            missing[k] += 1
for k in keys:
    vals = sorted(unique[k])
    print(f'KEY: {k}')
    print(f'  unique values count: {len(vals)}')
    print('  values:', vals)
    print(f'  missing in files: {missing[k]} / {len(chm_files)}')
    print()
