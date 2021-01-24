from decision_tree_node import TreeNode


printdict = lambda dic: [ print(key, ':', dic[key]) for key in dic]

class DecisionTree:
    def __init__(self):
        self.__root = None

    def get_tree_root(self):
        return self.__root

    def set_tree_root(self, node):
        if isinstance(node, TreeNode) and node.data['pos']=='root':
            self.__root = node
        else:
            raise TypeError('Wrong Node Type')

    def __test_node(self, tree_node, single_input_data):
        node_data = tree_node.data
        if node_data['type'] == 'leaf node right' or node_data['type'] == 'leaf node left':
            return node_data['class']
        else:
            if single_input_data[node_data['feature']] <= node_data['<']:
                tree_node = tree_node.get_left()
                res = self.__test_node(tree_node, single_input_data)
            else:
                tree_node = tree_node.get_right()
                res = self.__test_node(tree_node, single_input_data)
        return res

    def test(self, single_input_data, target):
        root_node = self.get_tree_root()
        res = self.__test_node(root_node, single_input_data)
        return (res, target)


    def score(self, dataset, target):

        test_res =[]
        for index in range(len(dataset)):
            test_res.append(self.test(dataset[index], target[index]))

        error = 0
        for pair in test_res:
            if pair[0] == pair[1]:
                error+=1
            else:
                continue

        score = (1 - error/len(dataset))*100
        return score

    def __traversal_pre_order_recursion(self, node):
        if node:
            printdict(node.data)
            print('*'*20)

            recursion = self.__traversal_pre_order_recursion(node.get_left())
            if recursion is None:
                recursion = self.__traversal_pre_order_recursion(node.get_right())
            return recursion
        else:
            return None

    def print_nodes(self):
        marker = self.__root
        if marker:
            self.__traversal_pre_order_recursion(marker)
        else:
            raise NotImplementedError("Empty Tree")


























