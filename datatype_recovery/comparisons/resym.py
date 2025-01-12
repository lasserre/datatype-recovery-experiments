import pandas as pd
from pathlib import Path
from typing import List, Tuple
import re
import json
from tqdm import tqdm

import dwarflib
from dwarflib import DwarfDebugInfo
from varlib.datatype import BuiltinType, DataType
from varlib import StructDatabase

def read_jsonl(jsonl_file:Path) -> List[dict]:
    with open(jsonl_file, 'r') as f:
        return [json.loads(line) for line in f.readlines()]

def extract_resym_vars(output:str) -> List[Tuple[str, str]]:
    '''Return a list of (name, type) tuples for variables in this "output" entry'''
    var_lines = output.split('\n')
    # each var string should be: "HexRays name: truth name, truth type"
    var_lines = [x if ',' in x else 'N/A, N/A' for x in var_lines]
    var_strings = [vs.split(':')[1].strip() if ':' in vs else vs.strip() for vs in var_lines]
    var_pieces = [x.split(',') for x in var_strings]
    return [(name.strip(), typestr.strip()) for name, typestr in var_pieces]

# from https://github.com/CMUSTRUDEL/DIRTY/blob/main/dirty/utils/code_processing.py#L10
def canonicalize_code(code):
    code = re.sub("//.*?\\n|/\\*.*?\\*/", "\\n", code, flags=re.S)
    lines = [l.rstrip() for l in code.split("\\n")]
    code = "\\n".join(lines)
    code = re.sub("@@\\w+@@(\\w+)@@\\w+", "\\g<1>", code)
    return code

def extract_rows_from_pred(pred:dict):
    proj = pred['proj']
    bin_name = pred['bin'][:6]
    func = pred['funname']
    canonical_code = canonicalize_code(pred['input'].split("```")[1])
    y_vars = extract_resym_vars(pred['output'])
    pred_vars = extract_resym_vars(pred['predict'])

    # try keeping only predictions that correspond to a true variable
    # (by list index, not name)

    # remove any extra preds
    pred_vars = pred_vars[:len(y_vars)]

    # pad any missing preds with N/A
    if len(y_vars) > len(pred_vars):
        pred_vars.extend([[None, None]]*(len(y_vars)-len(pred_vars)))

    for i in range(len(y_vars)):
        yield (
            proj, bin_name, func,
            y_vars[i][0],   # Name
            pred_vars[i][0],# NamePred
            y_vars[i][1],   # Type
            pred_vars[i][1],# Pred
            canonical_code
        )

def extract_predictions_df(vardecoder_preds:List[dict]) -> pd.DataFrame:
    '''
    Convert VarDecoder predictions (list of JSON dicts read from vardecoder_pred.jsonl) into
    a DataFrame format
    '''
    pred_df = pd.DataFrame([y for x in vardecoder_preds for y in extract_rows_from_pred(x)],
        columns=['Project','Binary','Function','Name','NamePred','Type','Pred','CanonicalCode'])

    # replace NaNs with void (they are incorrect, but we need to be able to parse them as types...
    # void should never be correct on its own here)
    assert len(pred_df[pred_df.Pred=='void']) == 0, 'void types already exist?'
    pred_df['Pred'] = pred_df.Pred.fillna('void')     # we have no 'void' entries as the type of a variable

    return pred_df

def read_vardecoder_data(resym_data:Path) -> Tuple[List[dict], List[dict]]:
    '''
    Return train_data, pred_data from the ReSym_data folder
    '''
    train_data = read_jsonl(resym_data/'vardecoder_train.jsonl')
    pred_data = read_jsonl(resym_data/'vardecoder_pred.jsonl')
    return (train_data, pred_data)

def update_type_mapping(ddi:DwarfDebugInfo, td_map:dict):
    tags_to_map = [
        'DW_TAG_base_type', 'DW_TAG_structure_type', 'DW_TAG_typedef',
        'DW_TAG_enumeration_type', 'DW_TAG_union_type',
        'DW_TAG_class_type'     # add classes to get FactoryImpl we were missing (treat as STRUCT)
    ]

    td_map.update({
        x.name: x.dtype.type_sequence_str
        for cu in ddi.dwarf.iter_CUs() for x in cu.iter_DIEs()
        if x.tag and x.tag in tags_to_map
    })

def get_typdef_map(tdmap_file:Path, binary_paths:List[Path]) -> dict:
    if tdmap_file.exists():
        print(f'Loading typedef map from file')
        with open(tdmap_file, 'r') as f:
            typedef_map = json.load(f)
    else:
        typedef_map = generate_typedef_map(tdmap_file, binary_paths)

    typedef_map['void'] = BuiltinType.from_standard_name('void').type_sequence_str
    return typedef_map

def generate_typedef_map(tdmap_file:Path, binary_paths:List[Path]):
    # FYI: took me about 20 min to generate this
    typedef_map = {}

    with dwarflib.UseStructDatabase(StructDatabase()):
        [update_type_mapping(DwarfDebugInfo.fromElf(bp, False), typedef_map) for bp in tqdm(binary_paths, desc='Building training data type map')]

    # save results to json
    with open(tdmap_file, 'w') as f:
        json.dump(typedef_map, f)

global_sdbs = {}    # map bin_name -> sdb
dwarf_func_data = [] # dwarf (bin_name, func_addr, func_name) entries for DataFrame


def extract_dwarf_functions(dwarf_csv:Path, test_binaries:List[Path]) -> pd.DataFrame:
    if dwarf_csv.exists():
        dwarf_func_df = pd.read_csv(dwarf_csv)
    else:
        # generate and save to csv
        for bp in tqdm(test_binaries, desc='Processing DWARF info from test split'):
            bin_name = bp.name[:6]      # take first 6 chars of hash (verified this is unique)
            global_sdbs[bin_name] = StructDatabase()

            with dwarflib.UseStructDatabase(global_sdbs[bin_name]):
                ddi = DwarfDebugInfo.fromElf(bp)
                dwarf_func_data.extend([(bin_name, k, v.name) for k, v in ddi.funcdies_by_addr.items()])

        dwarf_func_df = pd.DataFrame({
            'Binary': [x[0] for x in dwarf_func_data],
            'FunctionStart': [x[1] for x in dwarf_func_data],
            'Function': [x[2] for x in dwarf_func_data],
        })
        dwarf_func_df.to_csv(dwarf_csv, index=False)

    return dwarf_func_df

######################################################
# ReSym Type Preprocessing
######################################################

def remove_leading_str(dt:str, leading_str:str) -> str:
    return dt[len(leading_str):] if dt.startswith(leading_str) else dt

def remove_const(dt:str) -> str:
    return remove_leading_str(dt, 'const ')

def remove_struct(dt:str) -> str:
    return remove_leading_str(dt, 'struct ')

def remove_qualifiers(resym_type:str) -> str:
    '''Remove const/struct/etc..'''
    # processing in this order handles 'struct ', 'struct const', and 'const '
    # and we have all 3 in their dataset
    dt = remove_struct(resym_type)
    return remove_const(dt)

def resym_typestr_to_typeseq(typestr:str, typedef_map:dict) -> str:
    ltname, nptrs, arrdims = parse_type_str(typestr)

    # we only ever have arrays of pointers in the ReSym eval dataset
    # i.e. we only ever have: int* arr_p[5]
    # and not:                int (*p_arr)[5]

    ltype = typedef_map[ltname] if ltname in typedef_map else 'UNDEFINED'
    narrs = len(arrdims)

    return ','.join([*(['ARR']*narrs), *(['PTR']*nptrs), ltype])

# capture (type_name, ptr_expr, arr_expr)
def parse_type_str(type_str:str) -> Tuple[str, int, List[int]]:
    '''
    Parse a ReSym type string and return a tuple of:

    (leaf type name, number of pointer levels, list of array sizes)
    '''
    leaf_type, ptrs, arrs = re.match('([^\*\[]+)(\**)(.*)', type_str).groups()
    leaf_type = remove_qualifiers(leaf_type)
    return (leaf_type, len(ptrs), [int(x) for x in arrs[1:-1].split('][')] if arrs else '')