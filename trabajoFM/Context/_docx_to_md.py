#!/usr/bin/env python3
"""
Convert .docx to Markdown preserving headings, captions, and figure placeholders.
Based on the workflow from the 20260507 chat export.
"""
import json
from pathlib import Path
from docx import Document

def extract_docx_to_json(docx_path, json_path):
    """Extract paragraphs from DOCX to JSON."""
    doc = Document(docx_path)
    paragraphs = []
    for para in doc.paragraphs:
        paragraphs.append({
            "text": para.text,
            "style": para.style.name
        })
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(paragraphs, f, ensure_ascii=False, indent=2)
    print(f"Extracted {len(paragraphs)} paragraphs to {json_path}")

def convert_json_to_markdown(json_path, md_path):
    """Convert JSON to Markdown with heading levels and figure placeholders."""
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        paragraphs = json.load(f)
    
    md_lines = []
    for para in paragraphs:
        text = para['text'].strip()
        style = para['style']
        
        if not text:
            md_lines.append('')
            continue
        
        if style.startswith('Heading 1'):
            md_lines.append(f'# {text}')
        elif style.startswith('Heading 2'):
            md_lines.append(f'## {text}')
        elif style.startswith('Heading 3'):
            md_lines.append(f'### {text}')
        elif style.startswith('Heading 4'):
            md_lines.append(f'#### {text}')
        elif 'Caption' in style or text.startswith('Figure ') or text.startswith('Table '):
            # Figure/table captions as blockquote placeholders
            md_lines.append(f'> **[FIGURE PLACEHOLDER]** *{text}*')
        else:
            md_lines.append(text)
    
    md_content = '\n'.join(md_lines)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Wrote {len(md_content)} bytes to {md_path}")

if __name__ == '__main__':
    base_dir = Path(r'C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Granada\TrabajoFM\scripts\Python_Pipeline_SWAT_Pascal\swat_pipeline\trabajoFM\Context')
    
    # Merge intro, methods, and results into one Markdown file
    docx_files = [
        base_dir / 'intro_tmp_final_push.docx',
        base_dir / 'methods_tmp_final_push.docx',
        base_dir / 'results_tmp_final_push.docx'
    ]
    
    md_path = base_dir / 'TFM_final_content.md'
    all_md_lines = []
    
    for docx_path in docx_files:
        if not docx_path.exists():
            print(f"Skipping {docx_path.name} (not found)")
            continue
        
        json_path = base_dir / f'_temp_{docx_path.stem}.json'
        extract_docx_to_json(docx_path, json_path)
        
        # Load JSON and append to merged Markdown
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            paragraphs = json.load(f)
        
        for para in paragraphs:
            text = para['text'].strip()
            style = para['style']
            
            if not text:
                all_md_lines.append('')
                continue
            
            if style.startswith('Heading 1'):
                all_md_lines.append(f'# {text}')
            elif style.startswith('Heading 2'):
                all_md_lines.append(f'## {text}')
            elif style.startswith('Heading 3'):
                all_md_lines.append(f'### {text}')
            elif style.startswith('Heading 4'):
                all_md_lines.append(f'#### {text}')
            elif 'Caption' in style or text.startswith('Figure ') or text.startswith('Table '):
                all_md_lines.append(f'> **[FIGURE PLACEHOLDER]** *{text}*')
            else:
                all_md_lines.append(text)
        
        all_md_lines.append('\n---\n')  # Section separator
        json_path.unlink()  # cleanup temp JSON
    
    # Write merged Markdown
    md_content = '\n'.join(all_md_lines)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ Merged {len(docx_files)} DOCX files into {md_path} ({len(md_content)} bytes)")
