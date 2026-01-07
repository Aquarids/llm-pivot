import json
import re
from typing import Union, Dict, List

@staticmethod
def extract_json(response: str) -> Union[Dict, List]:
        
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"({[\s\S]*})",
        r"(\[[\s\S]*\])",
    ]
    
    json_str = None
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            json_str = matches[0].strip()
            break
    
    if not json_str:
        raise ValueError("No JSON found in response")
    
    def clean_json(s: str) -> str:
        s = re.sub(r'`[^`]*`', lambda m: m.group(0).replace('\n', ' ').replace('\r', ' ').replace('\t', ' '), s)
        s = s.replace('**', '').replace('__', '').replace('~~', '')
        s = s.replace('"', '"').replace('"', '"')
        s = s.replace(''', "'").replace(''', "'")
        s = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', s)
        s = re.sub(r',(\s*[}\]])', r'\1', s)
        s = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)
        
        def escape_in_strings(text):
            result = []
            in_string = False
            i = 0
            while i < len(text):
                if text[i] == '"' and (i == 0 or text[i-1] != '\\'):
                    in_string = not in_string
                    result.append(text[i])
                elif in_string:
                    if ord(text[i]) < 32 and text[i] not in '\n\r\t':
                        pass
                    else:
                        result.append(text[i])
                else:
                    result.append(text[i])
                i += 1
            return ''.join(result)
        
        s = escape_in_strings(s)
        return s

    def fix_escape_sequences(s: str) -> str:
        result = []
        i = 0
        in_string = False
        
        while i < len(s):
            char = s[i]
            
            if char == '"' and (i == 0 or s[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            if in_string and char == '\\' and i + 1 < len(s):
                next_char = s[i + 1]
                
                valid_escapes = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']
                
                if next_char in valid_escapes:
                    result.append(char)
                    result.append(next_char)
                    i += 2
                else:
                    result.append('\\\\')
                    i += 1
            else:
                result.append(char)
                i += 1
        
        return ''.join(result)
    
    strategies = [
        lambda s: s,
        lambda s: clean_json(s),
        lambda s: fix_escape_sequences(clean_json(s)),
        lambda s: fix_escape_sequences(s),
    ]
    
    last_error = None
    for strategy in strategies:
        try:
            processed = strategy(json_str)
            return json.loads(processed)
        except json.JSONDecodeError as e:
            last_error = e
            continue
    
    error_pos = getattr(last_error, 'pos', 0)
    context_start = max(0, error_pos - 100)
    context_end = min(len(json_str), error_pos + 100)
    
    raise ValueError(
        f"Failed to parse JSON: {str(last_error)}\n"
        f"Error position: {error_pos}\n"
        f"Context:\n{json_str[context_start:context_end]}\n"
        f"         {' ' * min(50, error_pos - context_start)}^"
    )


@staticmethod
def to_json(data):
    
    def json_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        else:
            try:
                return json.loads(json.dumps(obj))
            except:
                return str(obj)
    
    try:
        json_str = json.dumps(
            data,
            ensure_ascii=False,
            indent=4,
            default=json_serializable
        )
        
        formatted_str = (
            json_str
            .replace('    ', '  ')
            .replace('\\n', '\n')
            .replace('\\"', '"')
        )
        
        return formatted_str
        
    except Exception as e:
        raise e

@staticmethod
def compress_json(original_prompt):
    try:
        compressed = re.sub(
            r"```json\n([\s\S]*?)\n```",
            lambda m: "```json\n"
            + json.dumps(json.loads(m.group(1)), separators=(",", ":"))
            + "\n```",
            original_prompt,
        )

        compressed = re.sub(r"\n{3,}", "\n\n", compressed)
        compressed = re.sub(r"[ \t]{2,}", " ", compressed)
        return compressed.replace("\n\n", "\n")
    except Exception:
        return original_prompt
