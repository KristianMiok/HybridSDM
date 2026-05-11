#!/usr/bin/env python3
"""
Apply patch to add safety_identifier support to all generate_* scripts.

Run once after pulling the new code:
  python src/patch_safety_id.py
"""

import re
from pathlib import Path

SCRIPTS_TO_PATCH = [
    "generate_llm_trees_per_fold.py",
    "generate_llm_trees_v2.py",
    "generate_llm_trees.py",
]

OLD_CALL = '''def call_openai(prompt: str, temperature: float = 0.7) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert aquatic ecologist. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=16000,
    )
    return response.choices[0].message.content'''

NEW_CALL = '''import os as _os
SAFETY_IDENTIFIER = _os.environ.get("OPENAI_SAFETY_IDENTIFIER", "")

def call_openai(prompt: str, temperature: float = 0.7) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert aquatic ecologist. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 16000,
    }
    if SAFETY_IDENTIFIER:
        kwargs["safety_identifier"] = SAFETY_IDENTIFIER
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content'''


def patch_file(path: Path):
    if not path.exists():
        print(f"  Skip (not found): {path.name}")
        return False
    content = path.read_text()
    if "SAFETY_IDENTIFIER" in content:
        print(f"  Already patched: {path.name}")
        return False
    if OLD_CALL not in content:
        print(f"  ⚠ Pattern not found in: {path.name} (may have different format)")
        return False
    content = content.replace(OLD_CALL, NEW_CALL)
    path.write_text(content)
    print(f"  ✓ Patched: {path.name}")
    return True


if __name__ == "__main__":
    src_dir = Path(__file__).resolve().parent
    print(f"Patching scripts in {src_dir}/")
    for fname in SCRIPTS_TO_PATCH:
        patch_file(src_dir / fname)
    print("\nDone. Now set the safety identifier:")
    print('  export OPENAI_SAFETY_IDENTIFIER="$!b$1IFiDnLzAn"')
    print("\nThen run as normal:")
    print("  python src/generate_llm_trees_per_fold.py")
