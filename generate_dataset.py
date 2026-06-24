#!/usr/bin/env python3
"""
Harbour PRG/CH Dataset Generator for Fine-tuning qwen2.5-coder:14b
Generates structured JSONL dataset from .prg and .ch files with descriptions.
CLEANED VERSION - removes boilerplate, ensures code quality.
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configuration
HARBOUR_ROOT = Path("/home/fivetech/harbour")
OUTPUT_DIR = Path("/home/fivetech/finetune")
MAX_CODE_LENGTH = 8000
MIN_CODE_LENGTH = 80  # Minimum chars of actual code
TRAIN_RATIO = 0.9

# Module descriptions for contrib
MODULE_DESCRIPTIONS = {
    "hbhttpd": "Multithreaded HTTP/HTTPS server framework",
    "hbwin": "Windows API wrapper functions",
    "hbpgsql": "PostgreSQL database client library",
    "hbmysql": "MySQL database client library",
    "hbsqlit3": "SQLite3 database client library",
    "hbodbc": "ODBC database connectivity",
    "hbtip": "Internet protocol utilities (FTP, HTTP, SMTP, POP3)",
    "hbcurl": "libcurl wrapper for HTTP/FTP/SMTP operations",
    "hbssl": "OpenSSL wrapper for SSL/TLS encryption",
    "hbnf": "NanForum Toolkit - legacy Clipper compatibility functions",
    "hbct": "CA-Tools compatibility library",
    "hbmisc": "Miscellaneous utility functions",
    "hbgd": "Graphics drawing library (GD)",
    "hbcairo": "Cairo graphics library wrapper",
    "hbhpdf": "PDF generation library (libharu)",
    "hbbmp": "BMP image handling",
    "hbzebra": "Barcode generation library",
    "hbexpat": "XML parsing library (Expat)",
    "hbmxml": "XML generation library",
    "hbnetio": "Network I/O operations",
    "hbpipeio": "Process pipe I/O operations",
    "hbmemio": "Memory file I/O operations",
    "xhb": "Extended Harbour functions",
    "hbxpp": "xBase++ compatibility functions",
    "hbunix": "Unix-specific functions",
    "hbtpathy": "Telepath communication library",
    "hbblat": "Blat email sending utility",
    "hbblink": "Blinker function extender",
    "hbgs": "Ghostscript wrapper",
    "hbfship": "Fships library functions",
    "hbmzip": "ZIP file handling",
    "hbziparc": "ZIP archive handling",
    "hbxdiff": "File difference/patching",
    "hblzf": "LZF compression library",
    "hbmlzo": "LZO compression library",
    "hbbz2": "BZ2 compression library",
    "hbformat": "Text formatting utilities",
    "hbfoxpro": "FoxPro file format support",
    "hbplist": "Apple plist file format support",
    "hbcups": "CUPS printing system wrapper",
    "hbsms": "SMS sending via modem",
    "hbcomm": "Serial communication library",
    "hbfbird": "Firebird database client",
    "hbfimage": "FreeImage library wrapper",
    "hbdoc": "Documentation generation utilities",
    "hbtinymt": "Tiny Mersenne Twister PRNG",
    "hbtest": "Test framework utilities",
}


def remove_all_comments(code: str) -> str:
    """Remove ALL comments from Harbour code: //, /* */, *, and inline comments."""
    # Remove block comments /* ... */ (including multi-line)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # Remove single-line comments // ...
    code = re.sub(r'//[^\n]*', '', code)

    # Remove lines that are only star-prefixed comments: * ...
    # And standalone star lines used in block comment formatting
    lines = code.split("\n")
    result_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are ONLY comments (star-prefixed, standalone)
        if re.match(r'^\*\s', stripped) or stripped == '*' or stripped == '*/' or stripped == '/*':
            continue
        result_lines.append(line)
    code = "\n".join(result_lines)

    # Remove inline comments at end of lines: code  * comment
    # Pattern: something followed by whitespace then * ... (but not in strings)
    code = re.sub(r'([^\s"].*?)\s+\*[^"]*$', r'\1', code, flags=re.MULTILINE)

    return code


def clean_excessive_blank_lines(code: str) -> str:
    """Remove excessive blank lines (more than 2 consecutive)."""
    lines = code.split("\n")
    result_lines = []
    blank_count = 0

    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result_lines.append(line)
        else:
            blank_count = 0
            result_lines.append(line)

    return "\n".join(result_lines)


def remove_disabled_code(code: str) -> str:
    """Remove #if 0 ... #endif blocks (disabled code)."""
    lines = code.split("\n")
    result_lines = []
    in_disabled = False
    disabled_depth = 0

    for line in lines:
        stripped = line.strip()

        if stripped.upper().startswith("#IF 0") or stripped.upper().startswith("#IFDEF _HARBOUR_DISABLE"):
            in_disabled = True
            disabled_depth += 1
            continue

        if in_disabled:
            if stripped.upper().startswith("#ENDIF"):
                disabled_depth -= 1
                if disabled_depth <= 0:
                    in_disabled = False
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


def is_ch_file(filepath: Path) -> bool:
    """Check if file is a .ch (Clipper header) file."""
    return filepath.suffix.lower() == '.ch'


def has_real_ch_code(code: str) -> bool:
    """Check if .ch file contains actual preprocessor definitions."""
    upper = code.upper()

    # Must have at least one preprocessor construct
    ch_constructs = [
        "#XCOMMAND", "#XTRANSLATE", "#COMMAND", "#TRANSLATE",
        "#DEFINE", "#UNDEF", "#IFDEF", "#IFNDEF", "#IF ",
        "#INCLUDE", "#PRAGMA", "#ENDPROC",
    ]
    if not any(construct in upper for construct in ch_constructs):
        return False

    # Count actual definition lines
    lines = code.split("\n")
    def_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if stripped.startswith("#"):
            def_lines += 1

    return def_lines >= 3


def extract_ch_definitions(code: str) -> Dict:
    """Extract definitions from .ch file."""
    defines = []
    commands = []
    translates = []

    for line in code.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("#DEFINE ") or upper.startswith("#UNDEF "):
            parts = stripped.split()
            if len(parts) >= 2:
                defines.append(parts[1])
        elif upper.startswith("#XCOMMAND") or upper.startswith("#COMMAND"):
            commands.append(stripped[:60])
        elif upper.startswith("#XTRANSLATE") or upper.startswith("#TRANSLATE"):
            translates.append(stripped[:60])

    return {
        "defines": defines[:10],
        "commands": commands[:5],
        "translates": translates[:5]
    }


def generate_ch_description(filepath: Path, code: str) -> str:
    """Generate description for .ch file."""
    rel_path = filepath.relative_to(HARBOUR_ROOT)
    filename = filepath.stem

    desc_parts = []

    # Determine location context
    if rel_path.parts[0] == "include":
        desc_parts.append(f"Harbour header file: {rel_path}")
    elif rel_path.parts[0] == "contrib":
        module = rel_path.parts[1] if len(rel_path.parts) > 1 else ""
        module_desc = MODULE_DESCRIPTIONS.get(module, "")
        if module_desc:
            desc_parts.append(f"Header file for contribution module '{module}' ({module_desc}): {rel_path}")
        else:
            desc_parts.append(f"Header file for contribution module '{module}': {rel_path}")
    else:
        desc_parts.append(f"Harbour header file: {rel_path}")

    # Extract and describe definitions
    defs = extract_ch_definitions(code)

    if defs["defines"]:
        if len(defs["defines"]) <= 5:
            desc_parts.append(f"Defines constants: {', '.join(defs['defines'])}")
        else:
            desc_parts.append(f"Defines {len(defs['defines'])} constants including: {', '.join(defs['defines'][:5])}")

    if defs["commands"]:
        desc_parts.append(f"Declares {len(defs['commands'])} preprocessor commands")

    if defs["translates"]:
        desc_parts.append(f"Declares {len(defs['translates'])} preprocessor translations")

    # Common header purposes
    upper = code.upper()
    if "ES_" in upper or "EG_" in upper or "ERROR" in filename.upper():
        desc_parts.append("Defines error handling constants and codes")
    elif "INKEY" in upper or "K_" in upper:
        desc_parts.append("Defines keyboard input constants")
    elif "SET" in upper and "CH" in filename.upper():
        desc_parts.append("Defines SET command options")
    elif "COLOR" in upper or "_SET_" in upper:
        desc_parts.append("Defines color and display constants")
    elif "HB_" in upper or "HBEXT" in upper:
        desc_parts.append("Defines Harbour internal constants and macros")
    elif "THREAD" in upper:
        desc_parts.append("Defines threading constants and macros")
    elif "FILE" in upper or "F_" in upper:
        desc_parts.append("Defines file I/O constants")
    elif "DB" in upper or "RDD" in upper:
        desc_parts.append("Defines database/RDD constants")
    elif "COM" in upper or "SERIAL" in upper:
        desc_parts.append("Defines communication constants")
    elif "GT" in upper:
        desc_parts.append("Defines graphics terminal constants")
    elif "BOX" in upper or "BORDER" in upper:
        desc_parts.append("Defines box and border drawing constants")
    elif "MEMO" in upper:
        desc_parts.append("Defines memo field constants")

    return ". ".join(desc_parts)


def has_real_code(code: str) -> bool:
    """Check if code contains actual Harbour code (not just comments/includes)."""
    upper = code.upper()

    # Must have at least one of these code constructs
    code_constructs = [
        "FUNCTION ", "PROCEDURE ", "CREATE CLASS", "ENDCLASS",
        "METHOD ", "RETURN ", "LOCAL ", "MEMVAR ",
        "THREAD STATIC", "IF ", "FOR ", "WHILE ", "DO CASE",
        "BEGIN SEQUENCE", "SWITCH ", "REQUEST ",
        "INIT PROCEDURE", "EXIT PROCEDURE",
    ]
    if not any(construct in upper for construct in code_constructs):
        return False

    # Count actual code lines (non-empty, non-preprocessor, non-blank)
    lines = code.split("\n")
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        code_lines += 1

    return code_lines >= 8


def is_code_complete(code: str) -> bool:
    """Check if code is complete (proper ENDCLASS, balanced structures)."""
    upper = code.upper()

    # Check class definitions have matching ENDCLASS
    class_count = upper.count("CREATE CLASS")
    endclass_count = upper.count("ENDCLASS")
    if class_count > 0 and endclass_count < class_count:
        return False

    # Check DO CASE has ENDDO CASE
    docase_count = upper.count("DO CASE")
    endcase_count = upper.count("ENDCASE") + upper.count("END CASE")
    if docase_count > 0 and endcase_count < docase_count:
        return False

    # Check FOR/NEXT balance
    for_count = len(re.findall(r'\bFOR\s+\w+', upper))
    next_count = upper.count("\nNEXT") + (1 if upper.endswith("NEXT") else 0)
    # Be lenient - some code uses EXIT in loops

    # Check DO WHILE / ENDDO balance
    dowhile_count = upper.count("DO WHILE")
    enddo_count = upper.count("ENDDO") + upper.count("END DO")
    if dowhile_count > 0 and enddo_count < dowhile_count:
        return False

    # Check BEGIN SEQUENCE / END / RECOVER balance
    seq_count = upper.count("BEGIN SEQUENCE")
    end_count = upper.count("\nEND\n") + upper.count("\nEND ") + (1 if upper.endswith("\nEND") or upper.endswith(" END") else 0)

    # Check parentheses balance (lenient)
    open_p = code.count('(')
    close_p = code.count(')')
    if abs(open_p - close_p) > 3:
        return False

    # Check BEGIN/END blocks
    begin_count = len(re.findall(r'\bBEGIN\b', upper))
    end_block_count = len(re.findall(r'\bEND\b', upper)) - upper.count("ENDCLASS") - upper.count("ENDCASE") - upper.count("END IF") - upper.count("ENDDO")
    # Very lenient check - just ensure it's not wildly unbalanced
    if begin_count > 0 and end_block_count > begin_count + 5:
        return False

    return True


def extract_classes_and_functions(code: str) -> Dict:
    """Extract class and function definitions from code."""
    classes = []
    functions = []
    procedures = []

    for line in code.split("\n"):
        line_stripped = line.strip()
        upper = line_stripped.upper()

        # Class definitions
        if upper.startswith("CREATE CLASS"):
            parts = line_stripped.split()
            if len(parts) >= 3:
                class_name = parts[2]
                classes.append(class_name)

        # Function definitions
        if upper.startswith("FUNCTION ") or (upper.startswith("STATIC FUNCTION ")):
            parts = line_stripped.split()
            idx = 2 if upper.startswith("STATIC") else 1
            if len(parts) >= idx + 1:
                func_name = parts[idx].split("(")[0]
                functions.append(func_name)

        # Procedure definitions
        if upper.startswith("PROCEDURE ") or upper.startswith("STATIC PROCEDURE "):
            parts = line_stripped.split()
            idx = 2 if upper.startswith("STATIC") else 1
            if len(parts) >= idx + 1:
                proc_name = parts[idx].split("(")[0]
                procedures.append(proc_name)

        # INIT/EXIT procedures
        if upper.startswith("INIT PROCEDURE") or upper.startswith("EXIT PROCEDURE"):
            parts = line_stripped.split()
            if len(parts) >= 3:
                proc_name = parts[2].split("(")[0]
                procedures.append(proc_name)

    return {
        "classes": classes,
        "functions": functions,
        "procedures": procedures
    }


def categorize_file(filepath: Path) -> Tuple[str, str]:
    """Categorize a PRG file into category and subcategory."""
    rel_path = filepath.relative_to(HARBOUR_ROOT)
    parts = rel_path.parts

    if parts[0] == "src":
        if parts[1] == "rtl":
            return "rtl", categorize_rtl_file(filepath)
        elif parts[1] == "rdd":
            return "rdd", "rdd_core"
        elif parts[1] == "debug":
            return "rtl", "utility"
        else:
            return "rtl", "utility"
    elif parts[0] == "contrib":
        module = parts[1] if len(parts) > 1 else "unknown"
        return "contrib", categorize_contrib_module(module)
    elif parts[0] == "tests":
        return "tests", categorize_test_file(filepath)
    elif parts[0] == "utils":
        return "utils", categorize_utils_file(filepath)
    elif parts[0] == "extras":
        return "extras", categorize_extras_file(filepath)
    else:
        return "rtl", "utility"


def categorize_rtl_file(filepath: Path) -> str:
    """Categorize RTL files into subcategories."""
    name = filepath.stem.lower()

    if name.startswith("t") and not name.startswith("text"):
        if any(x in name for x in ["get", "browse", "column", "editor", "scalar", "object", "class"]):
            return "oop_class"
        elif any(x in name for x in ["menu", "popup", "topbar"]):
            return "ui_menu"
        elif any(x in name for x in ["check", "radio", "push", "list", "label", "button"]):
            return "ui_widget"
        elif any(x in name for x in ["edit", "memo"]):
            return "text_edit"
        elif any(x in name for x in ["persist", "profile", "symbol"]):
            return "oop_class"
        else:
            return "oop_class"
    elif "get" in name or "read" in name:
        return "get_system"
    elif any(x in name for x in ["err", "alert"]):
        return "error_handling"
    elif any(x in name for x in ["file", "dir", "ini", "type"]):
        return "file_io"
    elif any(x in name for x in ["db", "memo"]):
        return "database"
    else:
        return "utility"


def categorize_contrib_module(module: str) -> str:
    """Categorize contrib modules."""
    db_modules = {"hbpgsql", "hbmysql", "hbsqlit3", "hbodbc", "hbfbird", "rddsql", "rddpg",
                  "rddmy", "rddfb", "rddads", "rddbm", "rddmisc", "sddpg", "sddmy",
                  "sddoci", "sddodbc", "sddsqlt3", "sddfb", "rddado"}
    net_modules = {"hbtip", "hbcurl", "hbhttpd", "hbnetio", "hbcomio", "hbtcpio", "hbpipeio"}
    sec_modules = {"hbssl", "hbmagic"}
    gfx_modules = {"hbbmp", "hbcairo", "hbhpdf", "hbgd", "hbzebra", "hbfimage", "hbformat"}
    fmt_modules = {"hbexpat", "hbmxml", "hbfoxpro", "hbplist", "hbmemio"}
    plat_modules = {"hbwin", "hbunix", "hboslib", "gtalleg", "gtwvg", "gtwvw", "gtwvb"}
    compat_modules = {"hbnf", "hbct", "xhb", "hbxpp", "hbtpathy", "hbfship"}

    if module in db_modules:
        return "database"
    elif module in net_modules:
        return "network"
    elif module in sec_modules:
        return "security"
    elif module in gfx_modules:
        return "graphics"
    elif module in fmt_modules:
        return "data_format"
    elif module in plat_modules:
        return "platform"
    elif module in compat_modules:
        return "compatibility"
    else:
        return "utility"


def categorize_test_file(filepath: Path) -> str:
    """Categorize test files."""
    name = filepath.stem.lower()

    if any(x in name for x in ["class", "oob", "inherit", "scope", "data"]):
        return "oop"
    elif any(x in name for x in ["db", "rdd", "browse"]):
        return "database"
    elif any(x in name for x in ["speed", "bench"]):
        return "performance"
    elif any(x in name for x in ["str", "math", "date", "array", "for", "while", "if", "case",
                                   "static", "mem", "gt", "regex", "file", "err", "hello"]):
        return "language_basics"
    else:
        return "function_api"


def categorize_utils_file(filepath: Path) -> str:
    """Categorize utility files."""
    name = filepath.stem.lower()

    if "hbmk" in name or "build" in name:
        return "build_system"
    elif "test" in name or "rt_" in name:
        return "test_framework"
    elif "i18n" in name or "lang" in name:
        return "i18n"
    else:
        return "build_system"


def categorize_extras_file(filepath: Path) -> str:
    """Categorize extras files."""
    parts = filepath.relative_to(HARBOUR_ROOT).parts

    if len(parts) > 1:
        module = parts[1].lower()
        if "pdf" in module or "vpdf" in module:
            return "pdf"
        elif "xls" in module or "excel" in module:
            return "spreadsheet"
        elif "srv" in module or "http" in module:
            return "server"
    return "utility"


def generate_description(filepath: Path, code: str, category: str, subcategory: str) -> str:
    """Generate a comprehensive description for a PRG file."""
    rel_path = filepath.relative_to(HARBOUR_ROOT)
    module_name = rel_path.parts[1] if len(rel_path.parts) > 1 else "rtl"

    # Get module description if contrib
    module_desc = ""
    if category == "contrib" and module_name in MODULE_DESCRIPTIONS:
        module_desc = MODULE_DESCRIPTIONS[module_name]

    # Extract code elements
    elements = extract_classes_and_functions(code)

    # Generate description based on category and content
    desc_parts = []

    # File location context
    if category == "rtl":
        desc_parts.append(f"Harbour Runtime Library file: {rel_path}")
    elif category == "contrib":
        desc_parts.append(f"Harbour contribution module '{module_name}' ({module_desc}): {rel_path}")
    elif category == "tests":
        desc_parts.append(f"Harbour test program: {rel_path}")
    elif category == "utils":
        desc_parts.append(f"Harbour utility program: {rel_path}")
    elif category == "extras":
        desc_parts.append(f"Harbour extra library: {rel_path}")
    else:
        desc_parts.append(f"Harbour source file: {rel_path}")

    # Add code structure information
    if elements["classes"]:
        desc_parts.append(f"Defines classes: {', '.join(elements['classes'][:5])}")

    if elements["functions"]:
        if len(elements["functions"]) <= 5:
            desc_parts.append(f"Provides functions: {', '.join(elements['functions'])}")
        else:
            desc_parts.append(f"Provides {len(elements['functions'])} functions including: {', '.join(elements['functions'][:5])}")

    if elements["procedures"]:
        if len(elements["procedures"]) <= 3:
            desc_parts.append(f"Contains procedures: {', '.join(elements['procedures'])}")
        else:
            desc_parts.append(f"Contains {len(elements['procedures'])} procedures")

    # Add subcategory context
    subcategory_descriptions = {
        "oop_class": "This file implements object-oriented classes using Harbour's class system",
        "ui_widget": "This file defines UI widget classes for graphical interfaces",
        "ui_menu": "This file implements menu system classes",
        "text_edit": "This file provides text editing functionality",
        "get_system": "This file implements the GET system for input field handling",
        "scalar_type": "This file defines scalar type wrapper classes",
        "error_handling": "This file implements error handling and reporting",
        "file_io": "This file provides file I/O operations",
        "database": "This file handles database operations",
        "utility": "This file provides utility functions",
        "rdd_core": "This file implements core Record Driver Driver functionality",
        "rdd_driver": "This file implements a database driver",
        "network": "This file provides network protocol implementations",
        "security": "This file implements security and encryption functions",
        "graphics": "This file provides graphics and image processing capabilities",
        "data_format": "This file handles data format parsing and generation",
        "platform": "This file provides platform-specific functionality",
        "compatibility": "This file provides legacy compatibility functions",
        "language_basics": "This test file exercises basic Harbour language features",
        "function_api": "This test file tests specific function APIs",
        "oop": "This test file tests object-oriented programming features",
        "performance": "This test file benchmarks performance characteristics",
        "build_system": "This file is part of the build system tooling",
        "test_framework": "This file is part of the test framework",
        "i18n": "This file provides internationalization support",
        "pdf": "This file provides PDF generation capabilities",
        "spreadsheet": "This file provides spreadsheet generation capabilities",
        "server": "This file implements server functionality",
    }

    if subcategory in subcategory_descriptions:
        desc_parts.append(subcategory_descriptions[subcategory])

    return ". ".join(desc_parts)


def create_training_entry(filepath: Path, code: str, description: str) -> Dict:
    """Create a training entry in the instruction format."""
    return {
        "instruction": f"Write Harbour (xBase/Clipper) code for: {description}",
        "input": "",
        "output": code,
        "metadata": {
            "file_path": str(filepath.relative_to(HARBOUR_ROOT)),
            "language": "harbour",
            "description": description
        }
    }


def create_completion_entry(filepath: Path, code: str, description: str) -> Dict:
    """Create a completion-style training entry with diverse instructions."""
    # Generate diverse user prompts based on code content
    import random
    random.seed(hash(filepath))  # Deterministic per file

    elements = extract_classes_and_functions(code)
    upper = code.upper()

    # Different prompt templates based on content
    templates = []

    if elements["classes"]:
        templates.append(f"Implement the following Harbour classes: {', '.join(elements['classes'][:3])}. {description}")
        templates.append(f"Create Harbour OOP classes for the functionality described: {description}")

    if elements["functions"]:
        templates.append(f"Write Harbour functions: {', '.join(elements['functions'][:3])}. {description}")
        templates.append(f"Implement these Harbour functions: {description}")

    if elements["procedures"]:
        templates.append(f"Write a Harbour program with procedures: {', '.join(elements['procedures'][:3])}. {description}")

    if "#DEFINE" in upper or "#XCOMMAND" in upper:
        templates.append(f"Create Harbour preprocessor definitions: {description}")
        templates.append(f"Define Harbour macros and constants: {description}")

    # General templates
    templates.append(f"Write the following Harbour (xBase/Clipper) code:\n\n{description}")
    templates.append(f"Implement this Harbour module: {description}")
    templates.append(f"Here is a Harbour (xBase/Clipper) implementation:\n\n{description}")
    templates.append(f"Generate Harbour code for: {description}")

    # Select a random template
    user_prompt = random.choice(templates)

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Harbour (xBase/Clipper) programmer. Write clean, efficient code following Harbour conventions. Use proper Hungarian notation for variable names (c=character, n=numeric, l=logical, a=array, o=object, b=codeblock)."
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": code
            }
        ],
        "metadata": {
            "file_path": str(filepath.relative_to(HARBOUR_ROOT)),
            "language": "harbour",
            "description": description
        }
    }


def process_prg_file(filepath: Path) -> List[Dict]:
    """Process a single PRG file and generate training entries."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    # Skip empty files or very small files
    if len(code.strip()) < 50:
        return []

    # Step 1: Remove ALL comments (block, single-line, star-prefixed, inline)
    code = remove_all_comments(code)

    # Step 2: Remove disabled code blocks (#if 0)
    code = remove_disabled_code(code)

    # Step 3: Clean excessive blank lines
    code = clean_excessive_blank_lines(code)

    # Step 5: Strip leading/trailing whitespace
    code = code.strip()

    # Skip if no real code remains
    if not has_real_code(code):
        return []

    # Skip if code is too short
    if len(code) < MIN_CODE_LENGTH:
        return []

    # Skip if code is incomplete
    if not is_code_complete(code):
        return []

    # Truncate if too long
    if len(code) > MAX_CODE_LENGTH:
        lines = code.split("\n")
        truncated_lines = []
        current_length = 0
        for line in lines:
            if current_length + len(line) > MAX_CODE_LENGTH:
                break
            truncated_lines.append(line)
            current_length += len(line) + 1
        code = "\n".join(truncated_lines)

    # Categorize the file
    category, subcategory = categorize_file(filepath)

    # Generate description
    description = generate_description(filepath, code, category, subcategory)

    # Create training entry (chat format only - preferred for Qwen2.5-Coder)
    entry = create_completion_entry(filepath, code, description)
    entry["metadata"]["category"] = category
    entry["metadata"]["subcategory"] = subcategory

    return [entry]


def process_ch_file(filepath: Path) -> List[Dict]:
    """Process a single CH file and generate training entries."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    # Skip empty files
    if len(code.strip()) < 30:
        return []

    # Step 1: Remove ALL comments
    code = remove_all_comments(code)

    # Step 2: Remove disabled code blocks (#if 0)
    code = remove_disabled_code(code)

    # Step 3: Clean excessive blank lines
    code = clean_excessive_blank_lines(code)

    # Step 4: Strip leading/trailing whitespace
    code = code.strip()

    # Skip if no real code remains
    if not has_real_ch_code(code):
        return []

    # Skip if code is too short
    if len(code) < MIN_CODE_LENGTH:
        return []

    # Truncate if too long
    if len(code) > MAX_CODE_LENGTH:
        lines = code.split("\n")
        truncated_lines = []
        current_length = 0
        for line in lines:
            if current_length + len(line) > MAX_CODE_LENGTH:
                break
            truncated_lines.append(line)
            current_length += len(line) + 1
        code = "\n".join(truncated_lines)

    # Generate description
    description = generate_ch_description(filepath, code)

    # Determine category
    rel_path = filepath.relative_to(HARBOUR_ROOT)
    parts = rel_path.parts

    if parts[0] == "include":
        category = "include"
    elif parts[0] == "contrib":
        category = "contrib"
    elif parts[0] == "utils":
        category = "utils"
    elif parts[0] == "extras":
        category = "extras"
    else:
        category = "include"

    # Create training entry (chat format only)
    entry = create_completion_entry(filepath, code, description)
    entry["metadata"]["category"] = category
    entry["metadata"]["subcategory"] = "header"

    return [entry]


def main():
    """Main function to generate the dataset."""
    print("=" * 60)
    print("Harbour PRG/CH Dataset Generator (CLEANED)")
    print("=" * 60)

    # Find all PRG and CH files
    print("\n1. Finding all Harbour source files...")
    prg_files = list(HARBOUR_ROOT.rglob("*.prg"))
    ch_files = list(HARBOUR_ROOT.rglob("*.ch"))
    print(f"   Found {len(prg_files)} PRG files")
    print(f"   Found {len(ch_files)} CH files")

    # Process PRG files
    print("\n2. Processing PRG files...")
    all_entries = []
    category_counts = {}
    skipped_files = 0

    for i, filepath in enumerate(prg_files, 1):
        if i % 100 == 0:
            print(f"   Processing PRG file {i}/{len(prg_files)}...")

        entries = process_prg_file(filepath)
        if entries:
            all_entries.extend(entries)
            category = entries[0]["metadata"]["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        else:
            skipped_files += 1

    print(f"\n   PRG: Generated {len(all_entries)} entries, skipped {skipped_files} files")

    # Process CH files
    print("\n3. Processing CH files...")
    ch_entries = 0
    ch_skipped = 0

    for i, filepath in enumerate(ch_files, 1):
        if i % 20 == 0:
            print(f"   Processing CH file {i}/{len(ch_files)}...")

        entries = process_ch_file(filepath)
        if entries:
            all_entries.extend(entries)
            ch_entries += 1
            category = entries[0]["metadata"]["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        else:
            ch_skipped += 1

    print(f"\n   CH: Generated {ch_entries} file entries, skipped {ch_skipped} files")
    print(f"\n   Total: {len(all_entries)} training entries")

    # Print category statistics
    print("\n3. Category statistics:")
    for category, count in sorted(category_counts.items()):
        print(f"   {category}: {count} files")

    # Shuffle entries
    random.seed(42)
    random.shuffle(all_entries)

    # Split into train and validation
    print("\n4. Splitting into train/validation sets...")
    split_idx = int(len(all_entries) * TRAIN_RATIO)
    train_entries = all_entries[:split_idx]
    val_entries = all_entries[split_idx:]

    print(f"   Training set: {len(train_entries)} entries")
    print(f"   Validation set: {len(val_entries)} entries")

    # Save datasets
    print("\n5. Saving datasets...")

    # Save as JSONL (instruction format)
    train_jsonl_path = OUTPUT_DIR / "harbour_train.jsonl"
    val_jsonl_path = OUTPUT_DIR / "harbour_val.jsonl"

    with open(train_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            train_entry = {k: v for k, v in entry.items() if k != "metadata"}
            f.write(json.dumps(train_entry, ensure_ascii=False) + "\n")

    with open(val_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in val_entries:
            val_entry = {k: v for k, v in entry.items() if k != "metadata"}
            f.write(json.dumps(val_entry, ensure_ascii=False) + "\n")

    print(f"   Saved training JSONL: {train_jsonl_path}")
    print(f"   Saved validation JSONL: {val_jsonl_path}")

    # Save full dataset with metadata
    full_dataset_path = OUTPUT_DIR / "harbour_dataset_full.jsonl"
    with open(full_dataset_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"   Saved full dataset with metadata: {full_dataset_path}")

    # Generate statistics file
    total_files = len(prg_files) + len(ch_files)
    stats = {
        "total_prg_files": len(prg_files),
        "total_ch_files": len(ch_files),
        "total_files": total_files,
        "total_entries": len(all_entries),
        "skipped_prg": skipped_files,
        "skipped_ch": ch_skipped,
        "train_entries": len(train_entries),
        "val_entries": len(val_entries),
        "categories": category_counts,
        "files_per_category": {}
    }

    for entry in all_entries:
        cat = entry["metadata"]["category"]
        subcat = entry["metadata"]["subcategory"]
        if cat not in stats["files_per_category"]:
            stats["files_per_category"][cat] = {}
        stats["files_per_category"][cat][subcat] = stats["files_per_category"][cat].get(subcat, 0) + 1

    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"   Saved statistics: {stats_path}")

    # Generate README
    readme_content = f"""# Harbour Fine-tuning Dataset

## Overview
This dataset contains {len(all_entries)} training entries extracted from:
- {len(prg_files)} Harbour PRG (.prg) source files
- {len(ch_files)} Harbour Header (.ch) files

{skipped_files} PRG files and {ch_skipped} CH files were skipped due to quality issues.

## Dataset Format
The dataset is provided in JSONL format with the following structure:

### Instruction Format (harbour_train.jsonl / harbour_val.jsonl)
```json
{{"instruction": "...", "input": "", "output": "..."}}
```

### Full Dataset (harbour_dataset_full.jsonl)
```json
{{"instruction": "...", "input": "", "output": "...", "metadata": {{"file_path": "...", "language": "harbour", "category": "...", "subcategory": "..."}}}}
```

## Categories
- **include**: Header files with constants/macros ({category_counts.get('include', 0)} files)
- **rtl**: Harbour Runtime Library ({category_counts.get('rtl', 0)} files)
- **contrib**: Contribution libraries ({category_counts.get('contrib', 0)} files)
- **tests**: Test programs ({category_counts.get('tests', 0)} files)
- **utils**: Utility programs ({category_counts.get('utils', 0)} files)
- **extras**: Extra libraries ({category_counts.get('extras', 0)} files)

## Cleaning Applied
- Copyright/license headers removed
- Disabled code blocks (#if 0) removed
- Excessive trailing comments removed
- Excessive blank lines removed
- Files without actual code filtered out
- Incomplete code (missing ENDCLASS, etc.) filtered out

## Usage for Fine-tuning
```bash
# Using Ollama with Modelfile
FROM qwen2.5-coder:14b

# Training command
ollama create harbour-coder -f Modelfile

# Or use with other training frameworks
# The JSONL format is compatible with:
# - OpenAI fine-tuning API
# - Hugging Face transformers
# - Axolotl
# - LLaMA-Factory
```

## File Structure
- `harbour_train.jsonl` - Training set ({len(train_entries)} entries)
- `harbour_val.jsonl` - Validation set ({len(val_entries)} entries)
- `harbour_dataset_full.jsonl` - Full dataset with metadata
- `dataset_stats.json` - Dataset statistics
- `generate_dataset.py` - This script

## Source
The source files are from the Harbour project (https://harbour.github.io/),
an open-source Clipper-compatible compiler.
"""

    readme_path = OUTPUT_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"   Saved README: {readme_path}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
