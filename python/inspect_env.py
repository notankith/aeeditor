import os
p=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
print('env path:',p)
if os.path.exists(p):
    with open(p,'r',encoding='utf-8') as f:
        for i,l in enumerate(f,1):
            print(i, repr(l))
            if '=' in l:
                k,v = l.split('=',1)
                print(' parsed ->',k.strip(), '=>', v.strip()[:60])
else:
    print('no .env')
