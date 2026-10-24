import re
import os
import subprocess
from typing import List

# Common x86/x64 assembly opcodes whitelist/pattern
OPCODE_PATTERN = re.compile(r'^\s*([a-z]{2,8})\b', re.IGNORECASE)

def extract_opcodes_from_asm_file(asm_path: str) -> List[str]:
    """
    Extracts assembly opcodes from a disassembly text file or Kaggle .asm file.
    Filters out memory addresses, hexadecimal bytes, registers and labels.
    """
    opcodes = []
    # Known x86/x64 instructions
    VALID_OPCODES = {
        'mov', 'push', 'pop', 'call', 'sub', 'add', 'jmp', 'je', 'jne', 'jz', 'jnz',
        'lea', 'xor', 'cmp', 'test', 'ret', 'retn', 'nop', 'inc', 'dec', 'and', 'or',
        'shr', 'shl', 'sar', 'sal', 'imul', 'idiv', 'mul', 'div', 'int', 'leave',
        'setz', 'setnz', 'sete', 'setne', 'bt', 'btc', 'btr', 'bts', 'cld', 'std'
    }

    if not os.path.exists(asm_path):
        return opcodes

    with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Common IDA / objdump format: [Address] [Bytes] [Instruction] [Operands]
            parts = line.strip().split()
            for part in parts:
                cleaned = re.sub(r'[^a-zA-Z]', '', part).lower()
                if cleaned in VALID_OPCODES:
                    opcodes.append(cleaned)
                    break # Take the first instruction found on the line

    return opcodes


def extract_opcodes_using_objdump(binary_path: str) -> List[str]:
    """
    Disassembles executable binary using objdump utility (available on Linux / WSL / MinGW).
    Command equivalent to: objdump -d binary | awk ...
    """
    opcodes = []
    try:
        res = subprocess.run(
            ['objdump', '-d', binary_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        asm_pattern = re.compile(r'^\s*[0-9a-f]+:\s+([0-9a-f]{2}\s+)+\s*([a-z]+)', re.IGNORECASE)
        for line in res.stdout.splitlines():
            match = asm_pattern.search(line)
            if match:
                opcodes.append(match.group(2).lower())
    except Exception as e:
        print(f"Warning: objdump extraction failed for {binary_path}: {e}")

    return opcodes
