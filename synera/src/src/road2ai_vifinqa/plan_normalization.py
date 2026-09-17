"""Mechanical formatting/vectorization repair of model output; no financial rules."""
import ast
import copy
import json
import math
import re
import unicodedata

class Pairs(list):pass

def normalized_json(content):
    if '</think>' in content:content=content.split('</think>',1)[1]
    start=content.find('{')
    if start<0:raise ValueError('No JSON')
    try:
        root=json.JSONDecoder(object_pairs_hook=Pairs).raw_decode(content[start:])[0]
    except json.JSONDecodeError:
        # Some local constrained decoders occasionally insert a stray object or
        # bracket between otherwise complete step objects.  Recover only the
        # schema fields themselves; expressions are still parsed, validated and
        # executed by the normal compiler below.
        step_pattern=re.compile(
            r'"name"\s*:\s*"(p\d+)"\s*,\s*"expression"\s*:\s*("(?:\\.|[^"\\])*")',
            re.S)
        steps=[{'name':name,'expression':json.loads(expression)}
               for name,expression in step_pattern.findall(content[start:])]
        def string_field(name,default=None):
            match=re.search(r'"'+re.escape(name)+r'"\s*:\s*("(?:\\.|[^"\\])*")',content[start:],re.S)
            return json.loads(match.group(1)) if match else default
        missing_match=re.search(r'"missing_inputs"\s*:\s*(\[[^\]]*\])',content[start:],re.S)
        missing=json.loads(missing_match.group(1)) if missing_match else []
        result_step=string_field('result_step')
        if not steps or result_step is None:raise
        return {'steps':steps,'result_step':result_step,
                'expression_unit':string_field('expression_unit','requested'),
                'missing_inputs':missing}
    def convert(node):
        if isinstance(node,Pairs):
            keys=[k for k,v in node]
            if len(keys)!=len(set(keys)):
                if set(keys)!={'name','expression'}:raise ValueError('Ambiguous duplicate object keys')
                steps=[];current={}
                for key,value in node:
                    if key in current:
                        if set(current)!={'name','expression'}:raise ValueError('Incomplete duplicated step')
                        steps.append(current);current={}
                    current[key]=convert(value)
                if set(current)!={'name','expression'}:raise ValueError('Incomplete duplicated step')
                steps.append(current)
                return steps
            return {k:convert(v) for k,v in node}
        if isinstance(node,list):return [convert(v) for v in node]
        return node
    result=convert(root)
    result['steps']=[step for item in result.get('steps',[]) for step in (item if isinstance(item,list) else [item])]
    return result

def vectorize_assign(plan):
    """Only .assign(one_argument_lambda) is lowered to equivalent vector expressions."""
    class Replace(ast.NodeTransformer):
        def __init__(self,name,value):self.name=name;self.value=value
        def visit_Name(self,node):return copy.deepcopy(self.value) if node.id==self.name else node
    class Lower(ast.NodeTransformer):
        def visit_Subscript(self,node):
            node=self.generic_visit(node)
            # A fully specified MultiIndex .loc key followed by a column is a
            # scalar.  Local models often append ``.values[0]`` to that scalar.
            # Remove that suffix only for this syntactically unambiguous case.
            if not (isinstance(node.value,ast.Attribute) and node.value.attr=='values'
                    and isinstance(node.slice,ast.Constant) and node.slice.value==0):return node
            selected=node.value.value
            if not (isinstance(selected,ast.Subscript) and isinstance(selected.slice,ast.Constant)
                    and isinstance(selected.value,ast.Subscript)):return node
            loc_get=selected.value
            if not (isinstance(loc_get.value,ast.Attribute) and loc_get.value.attr=='loc'
                    and isinstance(loc_get.slice,ast.Tuple)):return node
            return selected
        def visit_Compare(self,node):
            node=self.generic_visit(node)
            # ``series == value & mask & mask`` is parsed as one comparison
            # against a nested bitwise tree. Flatten the tree and parenthesize
            # the first comparison. Genuine bit masks (non-constant first
            # operand) are left untouched.
            if len(node.ops)!=1 or len(node.comparators)!=1:return node
            right=node.comparators[0]
            operands=[];joins=[]
            def flatten(value):
                if isinstance(value,ast.BinOp) and isinstance(value.op,(ast.BitAnd,ast.BitOr)):
                    flatten(value.left);joins.append(value.op);flatten(value.right)
                else:operands.append(value)
            flatten(right)
            if len(operands)<2 or not isinstance(operands[0],ast.Constant) or len(joins)!=len(operands)-1:return node
            result=ast.Compare(left=node.left,ops=node.ops,comparators=[operands[0]])
            for join,operand in zip(joins,operands[1:]):
                result=ast.BinOp(left=result,op=join,right=operand)
            return result
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='merge' and not any(k.arg=='validate' for k in node.keywords):
                node.keywords.append(ast.keyword(arg='validate',value=ast.Constant(value='one_to_one')))
            if not (isinstance(node.func,ast.Attribute) and node.func.attr=='assign' and not node.args):return node
            if not any(isinstance(k.value,ast.Lambda) for k in node.keywords):return node
            base=node.func.value
            for kw in node.keywords:
                value=kw.value
                if isinstance(value,ast.Lambda):
                    a=value.args
                    if len(a.args)!=1 or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults:
                        raise ValueError('Unsupported lambda signature')
                    value=Replace(a.args[0].arg,base).visit(copy.deepcopy(value.body))
                base=ast.Call(func=ast.Attribute(value=base,attr='assign',ctx=ast.Load()),args=[],keywords=[ast.keyword(arg=kw.arg,value=value)])
            return base
    result=copy.deepcopy(plan)
    # Convert model step names to deterministic SSA names.  This repairs a
    # repeated temporary name without changing any dataframe/column reference.
    mapping={}
    renamed=[]
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            return ast.copy_location(ast.Name(id=mapping.get(node.id,node.id),ctx=node.ctx),node)
    for index,item in enumerate(result.get('steps',[])):
        expression=ast.unparse(ast.fix_missing_locations(Rename().visit(ast.parse(item['expression'],mode='eval'))))
        new_name=f's{index}'
        renamed.append({'name':new_name,'expression':expression})
        mapping[str(item['name'])]=new_name
    result['steps']=renamed
    if 'expression' in result:
        result['expression']=ast.unparse(ast.fix_missing_locations(Rename().visit(ast.parse(result['expression'],mode='eval'))))
    for item in result.get('steps',[])+[result]:
        tree=ast.parse(item['expression'],mode='eval')
        item['expression']=ast.unparse(ast.fix_missing_locations(Lower().visit(tree)))
    return result

def normalize_display_conversion(plan,question,frames):
    """Move a single final display-unit division to the compiler's unit boundary.

    Only the exact outer division matching the explicit question unit is removed.
    Internal arithmetic, rounding, and financial formulae are never rewritten.
    """
    plan=expand_question_constants(plan,question)
    from .compliant_query import output_contract, visible_frames
    from .expression_plan import inline_plan
    contract=output_contract(question)
    if not contract or contract['compiler_divides_by']==1:return plan
    visible=visible_frames(frames)
    source=inline_plan(plan.get('steps',[]),plan['expression'],frames=set(frames),
                       columns={c for d in visible.values() for c in d})
    root=ast.parse(source,mode='eval')
    node=root.body
    wrapper=None
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='float' and len(node.args)==1 and not node.keywords:
        wrapper=node;node=node.args[0]
    if not (isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.right,ast.Constant)
            and node.right.value==contract['compiler_divides_by']):return plan
    if wrapper is None:root.body=node.left
    else:wrapper.args[0]=node.left
    return {**plan,'steps':[],'expression':ast.unparse(ast.fix_missing_locations(root)),
            'expression_unit':'VND','normalization':'outer display conversion delegated to compiler'}

def expand_question_constants(plan,question):
    """Spell out units and percentage factors explicitly present in the question."""
    text=''.join(c for c in unicodedata.normalize('NFD',question.lower().replace('đ','d')) if unicodedata.category(c)!='Mn')
    candidates=[]
    units={'nghin ty':1e12,'tram ty':1e11,'ty':1e9,'trieu':1e6,'nghin':1e3,'ngan':1e3}
    for match in re.finditer(r'(\d+(?:[.,]\d+)?)\s*(nghin ty|tram ty|ty|trieu|nghin|ngan)\b',text):
        number=float(match.group(1).replace(',','.'));scale=units[match.group(2)]
        candidates.append((number*scale,f'({number!r} * {scale!r})'))
    for match in re.finditer(r'(\d+(?:[.,]\d+)?)\s*(?:%|phan tram)',text):
        number=float(match.group(1).replace(',','.'))
        candidates.extend([(number/100,f'({number!r} / 100)'),
                           (1+number/100,f'(1 + {number!r} / 100)'),
                           (1-number/100,f'(1 - {number!r} / 100)')])
    intrinsic={0,1,2,3,4,5,10,12,30,100,360,365,1000,1e6,1e9,1e11,1e12,0.5}
    class Expand(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,bool) or not isinstance(node.value,(int,float)) or node.value in intrinsic:return node
            for value,expression in candidates:
                if math.isclose(node.value,value,rel_tol=1e-13,abs_tol=1e-13):return ast.parse(expression,mode='eval').body
            return node
    if not candidates:return plan
    revised=copy.deepcopy(plan)
    for item in revised.get('steps',[])+[revised]:
        item['expression']=ast.unparse(ast.fix_missing_locations(Expand().visit(ast.parse(item['expression'],mode='eval'))))
    return revised
