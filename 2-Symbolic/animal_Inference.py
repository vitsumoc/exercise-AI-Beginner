# 向后推理专家系统示例

class Ask():
  # 默认情况下只有 y/n 但根据实际问题可以有多选
  def __init__(self,choices=['y','n']):
    self.choices = choices
  # 描述提问的方法
  def ask(self):
    # 当某个选项的长度大于1, 即不是 y/n 选项
    if max([len(x) for x in self.choices])>1:
      for i,x in enumerate(self.choices):
        # 添加序号格式化输出
        print("{0}. {1}".format(i,x),flush=True)
      x = int(input())
      # 保存被选中的值
      return self.choices[x]
     # 每个选项都只有一个字母, 在选项中加入'/', 输出
    else:
      print("/".join(self.choices),flush=True)
      return input()

# 对几种逻辑的抽象表达
class Content():
  def __init__(self,x):
    self.x=x
    
class If(Content):
  pass

class AND(Content):
  pass

class OR(Content):
  pass

# 规则实现
rules = {
  'default': Ask(['y','n']),
  'color' : Ask(['red-brown','black and white','other']),
  'pattern' : Ask(['dark stripes','dark spots']),
  'mammal': If(OR(['hair','gives milk'])),
  'carnivor': If(OR([AND(['sharp teeth','claws','forward-looking eyes']),'eats meat'])),
  'ungulate': If(['mammal',OR(['has hooves','chews cud'])]),
  'bird': If(OR(['feathers',AND(['flies','lies eggs'])])),
  'animal:monkey' : If(['mammal','carnivor','color:red-brown','pattern:dark spots']),
  'animal:tiger' : If(['mammal','carnivor','color:red-brown','pattern:dark stripes']),
  'animal:giraffe' : If(['ungulate','long neck','long legs','pattern:dark spots']),
  'animal:zebra' : If(['ungulate','pattern:dark stripes']),
  'animal:ostrich' : If(['bird','long nech','color:black and white','cannot fly']),
  'animal:pinguin' : If(['bird','swims','color:black and white','cannot fly']),
  'animal:albatross' : If(['bird','flies well'])
}

# 知识库实现
class KnowledgeBase():
  # 加载规则库, 制作一个空的memory
  def __init__(self,rules):
    self.rules = rules
    self.memory = {}
  
  # 获取信息输入的方法, 结合规则库推出需要提问的内容
  def get(self,name):
    # TODO
    if ':' in name:
      k,v = name.split(':')
      vv = self.get(k)
      return 'y' if v==vv else 'n'
    # 内存里有的问题，就不必再问
    if name in self.memory.keys():
      return self.memory[name]
    # 规则库中的每个key 都是 field
    for fld in self.rules.keys():
      if fld==name or fld.startswith(name+":"):
        # print(" + proving {}".format(fld))
        # value 赋值为y 或fld的后半部分
        value = 'y' if fld==name else fld.split(':')[1]
        # 执行并获得结果
        res = self.eval(self.rules[fld],field=name)
        # TODO
        if res!='y' and res!='n' and value=='y':
          self.memory[name] = res
          return res
        if res=='y':
          self.memory[name] = value
          return value
    # 内存中没有内容, 通过eval default
    res = self.eval(self.rules['default'],field=name)
    # 将结果加入内存
    self.memory[name]=res
    return res

  # eval expr表示询问表达式 field表示询问属性
  def eval(self,expr,field=None):
    # print(" + eval {}".format(expr))
    # 对于Ask，询问并获得结果
    if isinstance(expr,Ask):
      print(field)
      return expr.ask()
    # 对于If语句 执行其内容
    elif isinstance(expr,If):
      return self.eval(expr.x)
    # 对于 AND 或 list
    elif isinstance(expr,AND) or isinstance(expr,list):
      # 对于AND，将内容赋值给自己
      # 对于list，此处不处理
      expr = expr.x if isinstance(expr,AND) else expr
      # 遍历其中的每个条件并执行，并且只有在同时y时返回y
      for x in expr:
        if self.eval(x)=='n':
          return 'n'
      return 'y'
    elif isinstance(expr,OR):
      # 对于OR 执行其中的内容，任何y返回视为y
      for x in expr.x:
        if self.eval(x)=='y':
          return 'y'
      return 'n'
    # 如果传入的是字符串名称
    elif isinstance(expr,str):
      return self.get(expr)
    else:
      print("Unknown expr: {}".format(expr))

kb = KnowledgeBase(rules)
kb.get('animal')