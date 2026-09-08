import tokenize

def code_only(path):
    with open(path, 'rb') as f:
        toks = tokenize.tokenize(f.readline)
        return [(t.type, t.string) for t in toks
                if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                                  tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING)]

new = code_only('/mnt/d/Project/CacheBlend/example/blend.py')   # 加注释后的版本
old = code_only('/home/zzt/CacheBlend/example/blend.py')        # 加注释前的版本
if new == old:
    print("IDENTICAL: 代码逻辑完全一致，只有注释差异")
else:
    print("DIFFERENT: 代码有改动！")
    for i, (a, b) in enumerate(zip(new, old)):
        if a != b:
            print(f"  第 {i} 个 token: 新={a} 旧={b}")
