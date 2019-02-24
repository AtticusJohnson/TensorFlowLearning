class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None
        self.lastval = None



class SLinkedList:
    def __init__(self):
        self.headval = None

    # 插入在开头位置,链表左侧
    def add_left(self, newdata):
        NewNode = Node(newdata)
        NewNode.nextval = self.headval
        NewNode.nextval.lastval = NewNode
        NewNode.lastval = None
        self.headval = NewNode


    #在链表末尾添加一个新的node
    def add_right(self, newdata):
        NewNode = Node(newdata)
        if self.headval is None:
            self.headval = NewNode
            NewNode.lastval = None
            NewNode.nextval = None
            return

        laste = self.headval
        while(laste.nextval):
            laste = laste.nextval
        laste.nextval = NewNode
        NewNode.lastval = laste
        NewNode.nextval = None

    # 在中间的某个位置添加节点
    def insert(self, middle_node, newdata):
        if not middle_node:
            print("The mentioned node is absent")
            return

        NewNode = Node(newdata)
        middle_node.lastval.nextval = NewNode
        NewNode.nextval = middle_node

    # 移除某个节点
    def remove(self, remove_node):

        if remove_node.dataval == self.headval.dataval:
            self.headval = self.headval.nextval
        elif not remove_node.nextval.dataval:
            remove_node.dataval = None
        else:
            remove_node.lastval.nextval = remove_node.nextval





    # 打印链表
    def show(self):
        printval = self.headval
        while printval:
            print(printval.dataval)
            printval = printval.nextval


li = SLinkedList()
e1 = Node("Mon")
li.headval = e1
e2 = Node("Tue")
e3 = Node("Wed")

# 连接第一第二个节点
li.headval.nextval = e2
e2.lastval = li.headval
# 连接第二第三个节点
e3.lastval = e2
e2.nextval = e3


print(e2.nextval)
# 结果为e3内存地址<__main__.Node object at 0x0000001A0F9644BE0>

# 初始为："Mon" -> "Tue" -> "Wed"
# 对应节点：e1   ->  e2   ->  e3

li.add_left("Sun")  # 链表头插入"Sun"
li.add_right("Thu")  # 链表尾插入"Thu"
li.insert(e2, "error")  # 在"Tue"原来的位置插入"error"
li.remove(e1)  # 移除"Mon"
li.show()



