p = __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '.env'))
print('path:',p)
with open(p,'rb') as f:
    data = f.read()
print(repr(data))
print('len',len(data))
for i,b in enumerate(data[:200]):
    print(i, b)
