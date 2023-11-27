from gedcom.parser import Parser
from gedcom.element.individual import IndividualElement
from gedcom.element.family import FamilyElement

# 解析家谱文件
g = Parser()
g.parse_file('data/tsars.ged')

# 查看本体的基本内容
def show_info():
    # 所有成员
    d = g.get_element_dictionary()
    # v.get_name() 成员名称
    _l1 = [ (k,v.get_name()) for k,v in d.items() if isinstance(v,IndividualElement)]
    # print(_l1)
    d = g.get_element_dictionary()
    # 成员的child
    _l2 = [ (k,[x.get_value() for x in v.get_child_elements()]) for k,v in d.items() if isinstance(v,FamilyElement)]
    # print(_l2)

# show_info()

# 为本体添加内容
def add_on():
    # 加载族谱
    gedcom_dict = g.get_element_dictionary()
    # 单身与婚姻
    individuals, marriages = {}, {}

    # 元素转id
    def term2id(el):
        # pointer指的是 @1@ 这样的字符串, 起到id的作用
        return "i" + el.get_pointer().replace('@', '').lower()

    # 打开基础的 本体文件
    out = open("onto.ttl","a")

    # 遍历所有人
    for k, v in gedcom_dict.items():
        # 对于每个个人
        if isinstance(v,IndividualElement):
            # 子嗣和sibling各自集合
            children, siblings = set(), set()
            # 这个人的id
            idx = term2id(v)

            # title 就是姓名的结合 san zhang
            title = v.get_name()[0] + " " + v.get_name()[1]
            # 去除title中的特殊符号
            title = title.replace('"', '').replace('[', '').replace(']', '').replace('(', '').replace(')', '').strip()

            # FAMS 表示此人曾经构建的家庭 此人在家庭中是 HUSB 或 WIFE
            own_families = g.get_families(v, 'FAMS')
            for fam in own_families:
                # 此处可以记录此人的所有子嗣
                children |= set(term2id(i) for i in g.get_family_members(fam, "CHIL"))

            # FAMC 表示此人出生的家庭 此人在家庭中是 CHIL
            parent_families = g.get_families(v, 'FAMC')
            if len(parent_families):
                for member in g.get_family_members(parent_families[0], "CHIL"): # NB adoptive families i.e len(parent_families)>1 are not considered (TODO?)
                    if member.get_pointer() == v.get_pointer():
                        continue
                    # 此处可以记录此人所有的sibling
                    siblings.add(term2id(member))

            # 如果个人信息表存在此人, 则将上述数据补充
            if idx in individuals:
                children |= individuals[idx].get('children', set())
                siblings |= individuals[idx].get('siblings', set())
            # 在个人信息表中记录
            individuals[idx] = {'sex': v.get_gender().lower(), 'children': children, 'siblings': siblings, 'title': title}

        # 对于每个家庭
        elif isinstance(v,FamilyElement):
            wife, husb, children = None, None, set()
            # 获得所有的孩子
            children = set(term2id(i) for i in g.get_family_members(v, "CHIL"))

            try:
                # 如果能找到 WIFE 信息
                wife = g.get_family_members(v, "WIFE")[0]
                wife = term2id(wife)
                # 在个人表中补充她的孩子信息
                if wife in individuals: individuals[wife]['children'] |= children
                else: individuals[wife] = {'children': children}
            except IndexError: pass
            try:
                # 如果能找到 HUSB 信息
                husb = g.get_family_members(v, "HUSB")[0]
                husb = term2id(husb)
                # 在个人表中补充他的孩子信息
                if husb in individuals: individuals[husb]['children'] |= children
                else: individuals[husb] = {'children': children}
            except IndexError: pass

            # 在婚姻表中记录夫妻信息
            if wife and husb: marriages[wife + husb] = (term2id(v), wife, husb)

    # 遍历整理后的个人信息表
    for idx, val in individuals.items():
        # 准备凭贴字符串
        added_terms = ''
        if val['sex'] == 'f':
            parent_predicate, sibl_predicate = "isMotherOf", "isSisterOf"
        else:
            parent_predicate, sibl_predicate = "isFatherOf", "isBrotherOf"
        if len(val['children']):
            added_terms += " ;\n    fhkb:" + parent_predicate + " " + ", ".join(["fhkb:" + i for i in val['children']])
        if len(val['siblings']):
            added_terms += " ;\n    fhkb:" + sibl_predicate + " " + ", ".join(["fhkb:" + i for i in val['siblings']])
        # 记录关系 兄弟姐妹/上下两代
        out.write("fhkb:%s a owl:NamedIndividual, owl:Thing%s ;\n    rdfs:label \"%s\" .\n" % (idx, added_terms, val['title']))

    # 遍历婚姻表, 记录婚姻关系
    for k, v in marriages.items():
        out.write("fhkb:%s a owl:NamedIndividual, owl:Thing ;\n    fhkb:hasFemalePartner fhkb:%s ;\n    fhkb:hasMalePartner fhkb:%s .\n" % v)

    # 有特征的成员 (也就是后续可以被我们用来推断的成员)
    out.write("[] a owl:AllDifferent ;\n    owl:distinctMembers (")
    for idx in individuals.keys():
        out.write("    fhkb:" + idx)
    for k, v in marriages.items():
        out.write("    fhkb:" + v[0])
    out.write("    ) .")
    out.close()

# add_on()

import rdflib
from owlrl import DeductiveClosure, OWLRL_Extension

# 使用rdflib打开本体文件
g = rdflib.Graph()
g.parse("onto.ttl", format="turtle")

# 查看其中的三元组数量
print("Triplets found:%d" % len(g))

# 这里对三元组的数量进行了拓展 (对于我们之前选出的人 以及之前选出的亲属关系)
DeductiveClosure(OWLRL_Extension).expand(g)
print("Triplets after inference:%d" % len(g))

# 可以使用 SPARQL 查询亲属关系了, 例如查询所有的 uncle
qres = g.query(
    """SELECT DISTINCT ?aname ?bname
       WHERE {
          ?a fhkb:isUncleOf ?b .
          ?a rdfs:label ?aname .
          ?b rdfs:label ?bname .
       }""")

for row in qres:
    print("%s is uncle of %s" % row)