# Developed by Chris Young in 2019
from Decision_Regression_Tree_Node import TreeNode
from copy import deepcopy

# print dictionary data
printdict = lambda dic: [print(key, ':', dic[key]) for key in dic]


# classification tree
class DecisionTree:
    def __init__(self):
        self.__root = None
        self.tree_depth = 0
        self.leaf_node_count = 0

    def get_tree_root(self):
        return self.__root

    def set_tree_root(self, node):
        if isinstance(node, TreeNode):
            self.__root = node
        else:
            raise TypeError('Wrong Node Type')

    def __test_node(self, tree_node, single_input_data):
        node_data = tree_node.data
        if node_data['type'] == 'leaf node':
            return node_data['class']
        elif (node_data['type'] == 'root node' and ('feature' not in node_data)):
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

    def test_notarget(self, single_input_data):
        root_node = self.get_tree_root()
        res = self.__test_node(root_node, single_input_data)
        return res

    def classify_ensemble(self, single_input_data):
        return self.test_notarget(single_input_data)

    def score(self, dataset, target):
        dataset = deepcopy(dataset)
        target = deepcopy(target)
        test_res = []
        for index in range(len(dataset)):
            test_res.append(self.test(dataset[index], target[index]))

        error = 0
        for pair in test_res:
            if pair[0] != pair[1]:
                error += 1
            else:
                continue

        score = (1 - error / len(dataset)) * 100
        return score

    def __traversal_pre_order_recursion(self, node):
        if node:
            printdict(node.data)
            print('*' * 20)

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

    # build a tree
    def build_tree(self, data_train, data_train_target, max_depth=None, multiclass=2):
        data_train = deepcopy(data_train)
        data_train_target = deepcopy(data_train_target)
        data_set = self.__merge_data(data_train, data_train_target)
        res = self.__split_recursion(data_set, multiclass, max_depth, )
        if res is not None:
            root_node = res[0]
            root_node.data['type'] = 'root node'
            # decision_tree = DecisionTree()
            self.set_tree_root(root_node)
        else:
            self.set_tree_root(None)

    # merget the data & target
    def __merge_data(self, data_set, data_set_target):
        data_merge = []
        for res in zip(data_set, data_set_target):
            data_merge.append(res[0] + [res[1]])
        return data_merge

    ## split_the dataset repeatedly in a recursion and a left node & right node
    ## with all child nodes connected
    def __split_recursion(self, data_set, multiclass, max_depth=None, split_info=None):
        if max_depth is not None and max_depth <= 0:
            raise NotImplementedError('Tree Depth <=0')
            return None

        split_info = self.__get_split_info(data_set, multiclass)
        left_set, right_set, split_info = self.__split_data(data_set, split_info)
        self.tree_depth += 1

        # left branch can be split but right branch cannot
        if type(left_set) is not str and type(right_set) is str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth is not None else None
                res = self.__split_recursion(left_set, multiclass, max_depth, split_info)
                if len(res) > 0:
                    child_left_node = res[0]
                    current_split_info = deepcopy(split_info)

                    mother_node_data = {
                        'type': 'non-leaf node',
                        '<': current_split_info['<'],
                        'gini': current_split_info['gini'],
                        'feature': current_split_info['feature'],
                        'sample': current_split_info['sample'],
                        'value': current_split_info['value'],
                        'lr_gini': current_split_info['lr_gini'],
                        'lr_class': current_split_info['lr_class'],
                    }

                    child_right_node_data = {
                        'type': 'leaf node',
                        'gini': current_split_info['lr_gini'][1],
                        'sample': current_split_info['value'][1],
                        'value': current_split_info['lr_class'][1],
                        'class': current_split_info['lr_class'][1].index(max(current_split_info['lr_class'][1])),
                        'pos': 'right',
                    }
                    self.leaf_node_count += 1

                    child_left_node.data['pos'] = 'left'

                mother_node = TreeNode(mother_node_data)
                child_right_node = TreeNode(child_right_node_data)
                mother_node.set_left(child_left_node)
                mother_node.set_right(child_right_node)

                return (mother_node, split_info)

            else:
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': split_info['<'],
                    'gini': split_info['gini'],
                    'feature': split_info['feature'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'lr_gini': split_info['lr_gini'],
                    'lr_class': split_info['lr_class'],
                }

                left_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_class'][0],
                    'class': split_info['lr_class'][0].index(max(split_info['lr_class'][0])),
                    'pos': 'left',
                }
                self.leaf_node_count += 1

                right_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_class'][1],
                    'class': split_info['lr_class'][1].index(max(split_info['lr_class'][1])),
                    'pos': 'right',
                }
                self.leaf_node_count += 1

                mother_node = TreeNode(mother_node_data)
                left_node = TreeNode(left_node_data)
                right_node = TreeNode(right_node_data)
                mother_node.set_left(left_node)
                mother_node.set_right(right_node)

                return (mother_node, split_info)

        # left branch cannot be split but right branch can
        elif type(left_set) is str and type(right_set) is not str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth != None else None
                res = self.__split_recursion(right_set, multiclass, max_depth, split_info)
                if len(res) > 0:
                    child_right_node = res[0]
                    current_split_info = deepcopy(split_info)

                    mother_node_data = {
                        'type': 'non-leaf node',
                        '<': current_split_info['<'],
                        'gini': current_split_info['gini'],
                        'feature': current_split_info['feature'],
                        'sample': current_split_info['sample'],
                        'value': current_split_info['value'],
                        'lr_gini': current_split_info['lr_gini'],
                        'lr_class': current_split_info['lr_class'],
                    }

                    child_left_node_data = {
                        'type': 'leaf node',
                        'gini': current_split_info['lr_gini'][0],
                        'sample': current_split_info['value'][0],
                        'value': current_split_info['lr_class'][0],
                        'class': current_split_info['lr_class'][0].index(max(current_split_info['lr_class'][0])),
                        'pos': 'left',
                    }
                    self.leaf_node_count += 1

                    child_right_node.data['pos'] = 'right'

                    mother_node = TreeNode(mother_node_data)
                    child_left_node = TreeNode(child_left_node_data)
                    mother_node.set_left(child_left_node)
                    mother_node.set_right(child_right_node)

                    return (mother_node, split_info)

            else:
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': split_info['<'],
                    'gini': split_info['gini'],
                    'feature': split_info['feature'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'lr_gini': split_info['lr_gini'],
                    'lr_class': split_info['lr_class'],
                }

                left_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_class'][0],
                    'class': split_info['lr_class'][0].index(max(split_info['lr_class'][0])),
                    'pos': 'left',
                }
                self.leaf_node_count += 1

                right_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_class'][1],
                    'class': split_info['lr_class'][1].index(max(split_info['lr_class'][1])),
                    'pos': 'right',
                }
                self.leaf_node_count += 1

                mother_node = TreeNode(mother_node_data)
                left_node = TreeNode(left_node_data)
                right_node = TreeNode(right_node_data)
                mother_node.set_left(left_node)
                mother_node.set_right(right_node)

                return (mother_node, split_info)

        # both branches can be split
        elif type(left_set) is not str and type(right_set) is not str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth is not None else None

                res_left = self.__split_recursion(left_set, multiclass, max_depth, split_info, )
                if len(res_left) > 0:
                    left_child_node = res_left[0]
                    recursion_split_info_left = res_left[-1]

                res_right = self.__split_recursion(right_set, multiclass, max_depth, split_info)
                if len(res_right) > 0:
                    right_child_node = res_right[0]
                    recursion_split_info_right = res_right[-1]

                current_split_info = split_info

                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': current_split_info['<'],
                    'gini': current_split_info['gini'],
                    'feature': current_split_info['feature'],
                    'sample': current_split_info['sample'],
                    'value': current_split_info['value'],
                    'lr_gini': current_split_info['lr_gini'],
                    'lr_class': current_split_info['lr_class'],
                }

                mother_node = TreeNode(mother_node_data)
                mother_node.set_left(left_child_node)
                mother_node.set_right(right_child_node)

                return (mother_node, split_info)

            else:
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': split_info['<'],
                    'gini': split_info['gini'],
                    'feature': split_info['feature'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'lr_gini': split_info['lr_gini'],
                    'lr_class': split_info['lr_class'],
                }

                left_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_class'][0],
                    'class': split_info['lr_class'][0].index(max(split_info['lr_class'][0])),
                    'pos': 'left',
                }
                self.leaf_node_count += 1

                right_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_class'][1],
                    'class': split_info['lr_class'][1].index(max(split_info['lr_class'][1])),
                    'pos': 'right',
                }
                self.leaf_node_count += 1

                mother_node = TreeNode(mother_node_data)
                left_node = TreeNode(left_node_data)
                right_node = TreeNode(right_node_data)
                mother_node.set_left(left_node)
                mother_node.set_right(right_node)

                return (mother_node, split_info)

        # both branches cannot be split
        elif type(left_set) is str and type(right_set) is str:
            if split_info['type'] == 'leaf node':
                leaf_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['gini'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'class': split_info['lr_class'].index(max(split_info['lr_class'])),
                }
                self.leaf_node_count += 1
                leaf_node = TreeNode(leaf_node_data)
                return (leaf_node, split_info)

            elif split_info['type'] == 'non-leaf node':
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': split_info['<'],
                    'gini': split_info['gini'],
                    'feature': split_info['feature'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'lr_gini': split_info['lr_gini'],
                    'lr_class': split_info['lr_class'],
                }

                left_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_class'][0],
                    'class': split_info['lr_class'][0].index(max(split_info['lr_class'][0])),
                    'pos': 'left',
                }
                self.leaf_node_count += 1

                right_node_data = {
                    'type': 'leaf node',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_class'][1],
                    'class': split_info['lr_class'][1].index(max(split_info['lr_class'][1])),
                    'pos': 'right',
                }
                self.leaf_node_count += 1

                mother_node = TreeNode(mother_node_data)
                left_node = TreeNode(left_node_data)
                right_node = TreeNode(right_node_data)
                mother_node.set_left(left_node)
                mother_node.set_right(right_node)

                return (mother_node, split_info)

    # calculate gini of all features of dataset to get the split feature info
    # data_set must have data & target
    # multiclass dataset target must modify as 0,1,2,3...
    def __get_split_info(self, data_set, multiclass_number=2):
        if data_set != 'True' and data_set != 'False':
            feature_gini_lst = []
            for col_pos in range(len(data_set[0]) - 1):
                feature_gini_lst.append(self.__calc_feature_gini(data_set, col_pos, multiclass=multiclass_number))
                if feature_gini_lst[-1] is None:
                    raise NotImplementedError('get_split_info func error: feature_gini is None...')
                    break

        else:
            return data_set
        return sorted(feature_gini_lst, key=lambda val: val['gini'])[0]

    # split dataset based on the split_info for one time
    def __split_data(self, dataset, split_info):
        left_set = []
        right_set = []

        col_pos = split_info['feature']
        if col_pos is not None:
            col_min = [min(val) for val in list(zip(*dataset))][col_pos]
            col_max = [max(val) for val in list(zip(*dataset))][col_pos]

        if split_info['lr_gini'][0] == 0 and split_info['lr_gini'][1] != 0:
            left_set = 'pure'
            for index in range(len(dataset)):
                if split_info['<'] == col_min:
                    if dataset[index][split_info['feature']] > split_info['<']:
                        right_set.append(dataset[index])
                else:
                    if dataset[index][split_info['feature']] >= split_info['<']:
                        right_set.append(dataset[index])

        elif split_info['lr_gini'][1] == 0 and split_info['lr_gini'][0] != 0:
            right_set = 'pure'
            for index in range(len(dataset)):
                if split_info['<'] == col_max:
                    if dataset[index][split_info['feature']] < split_info['<']:
                        left_set.append(dataset[index])
                else:
                    if dataset[index][split_info['feature']] <= split_info['<']:
                        left_set.append(dataset[index])

        elif split_info['lr_gini'][0] == 0 and split_info['lr_gini'][1] == 0:
            left_set = 'pure'
            right_set = 'pure'

        else:
            for index in range(len(dataset)):

                if split_info['<'] != col_min and split_info['<'] != col_max:
                    if dataset[index][split_info['feature']] <= split_info['<']:
                        left_set.append(dataset[index])
                    else:
                        right_set.append(dataset[index])

                elif split_info['<'] == col_min:
                    if dataset[index][split_info['feature']] == split_info['<']:
                        left_set.append(dataset[index])
                    else:
                        right_set.append(dataset[index])

                elif split_info['<'] == col_max:
                    if dataset[index][split_info['feature']] == split_info['<']:
                        right_set.append(dataset[index])
                    else:
                        left_set.append(dataset[index])

                elif split_info['<'] == col_max and split_info['<'] == col_min:
                    raise NotImplementedError('only one value in data split function...')

        return [left_set, right_set, split_info]

    # calculate gini for one feature for multiclass
    # return a split gini point
    # [total_gini, middle_value, feature_position, value_index, left_count, right_count]
    # multiclass target: 0, 1, 2, 3, 4,....
    # must modify the dataset target: 0,1,2,3,4....
    def __calc_feature_gini(self, data_set, pos, multiclass=2):
        gini_lst = []
        data_sorted = deepcopy(sorted(data_set, key=lambda row: row[pos]))
        # data_sorted = deepcopy(data_set)
        col_max = [max(val) for val in list(zip(*data_set))][pos]
        col_min = [min(val) for val in list(zip(*data_set))][pos]

        if col_max > col_min:
            for index in range(len(data_sorted)):

                left_count = [0] * multiclass  # left branch: [0]->yes, [1]->no
                right_count = [0] * multiclass  # right branch: [0]->yes, [1]->no

                if index + 1 == len(data_sorted):
                    break
                else:
                    # calc middle value between certain feature data
                    mid_val = (data_sorted[index][pos] + data_sorted[index + 1][pos]) / 2

                    # count class number around this middle value
                    for each in data_sorted:
                        if (each[pos] < mid_val) if mid_val != col_min else each[pos] <= mid_val:
                            for target in range(multiclass):
                                if each[-1] == target:
                                    left_count[target] += 1
                                else:
                                    continue

                        elif (each[pos] > mid_val) if mid_val != col_max else each[pos] >= mid_val:
                            for target in range(multiclass):
                                if each[-1] == target:
                                    right_count[target] += 1
                                else:
                                    continue

                        elif each[pos] == mid_val:
                            for target in range(multiclass):
                                if each[-1] == target:
                                    left_count[target] += 1
                                else:
                                    continue

                # the branches are not empty
                if sum(left_count) != 0 and sum(right_count) != 0:
                    sum_left = 0
                    sum_right = 0
                    for target in range(multiclass):
                        sum_left += (left_count[target] / sum(left_count)) ** 2
                        sum_right += (right_count[target] / sum(right_count)) ** 2
                    left_gini = 1 - sum_left
                    right_gini = 1 - sum_right

                else:
                    print('Error: division zero,', 'feature position:', pos, 'index:', index)
                    return None

                total_count = sum(left_count) + sum(right_count)
                total_gini = (sum(left_count) / total_count) * left_gini + (sum(right_count) / total_count) * right_gini
                feature_position = pos

                result = {
                    '<': mid_val,
                    'type': 'non-leaf node',
                    "gini": total_gini,
                    'feature': feature_position,
                    'sample': total_count,
                    'value': [sum(left_count), sum(right_count)],
                    'lr_class': [left_count, right_count],
                    'lr_gini': [left_gini, right_gini],
                }

                gini_lst.append(result)

        elif col_max == col_min:
            mid_val = (col_max + col_min) / 2
            class_count = [0] * multiclass

            # count the target
            for data in data_sorted:
                for target in range(multiclass):
                    if data[-1] == target:
                        class_count[target] += 1
                    else:
                        continue

            # calc gini impurity
            gini_sum = 0
            for val in class_count:
                gini_sum += ((val / sum(class_count)) ** 2)

            gini_impurity = 1 - gini_sum

            result = {
                '<': mid_val,
                'type': 'leaf node',
                "gini": gini_impurity,
                'feature': None,
                'sample': sum(class_count),
                'value': class_count,
                'lr_class': class_count,
                'lr_gini': [0, 0],
            }

            gini_lst.append(result)

        return sorted(gini_lst, key=lambda val: val['gini'])[0]


# regression tree
class RegressionTree:
    def __init__(self):
        self.__root = None
        self.tree_depth = 0
        self.leaf_node_count = 0

    def get_tree_root(self):
        return self.__root

    def set_tree_root(self, node):
        if isinstance(node, TreeNode):
            self.__root = node
        else:
            raise TypeError('Wrong Node Type')

    def __test_node(self, tree_node, single_input_data):
        node_data = tree_node.data
        if node_data['type'] == 'leaf node':
            return node_data['class']
        elif (node_data['type'] == 'root node' and ('feature' not in node_data)):
            return node_data['class']
        else:
            if single_input_data[node_data['feature']] <= node_data['<']:
                tree_node = tree_node.get_left()
                res = self.__test_node(tree_node, single_input_data)
            else:
                tree_node = tree_node.get_right()
                res = self.__test_node(tree_node, single_input_data)
        return res

    def test_target(self, single_input_data, target):
        root_node = self.get_tree_root()
        res = self.__test_node(root_node, single_input_data)
        return (res, target)

    def test(self, single_input_data):
        root_node = self.get_tree_root()
        res = self.__test_node(root_node, single_input_data)
        return res

    def error(self, dataset, target):
        dataset = deepcopy(dataset)
        target = deepcopy(target)
        test_res = []
        for index in range(len(dataset)):
            test_res.append(self.test_target(dataset[index], target[index]))

        error = 0
        for pair in test_res:
            error += abs(pair[0] - pair[1]) / len(dataset)
        return error

    def __traversal_pre_order_recursion(self, node):
        if node:
            printdict(node.data)
            print('*' * 20)
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

    # build a tree
    def build_tree(self, data_train, data_train_target, max_depth=None, max_estimator=None):
        data_train = deepcopy(data_train)
        data_train_target = deepcopy(data_train_target)
        data_set = self.__merge_data(data_train, data_train_target)
        # split dataset and get the root node
        res = self.__split_recursion(data_set, max_depth, max_estimator)
        if res is not None:
            root_node = res[0]
            root_node.data['type'] = 'root node'
            self.set_tree_root(root_node)
        else:
            self.set_tree_root(None)

    # merget the data & target
    def __merge_data(self, data_set, data_set_target):
        data_merge = []
        for res in zip(data_set, data_set_target):
            data_merge.append(res[0] + [res[1]])
        return data_merge

    # split_the dataset repeatedly in a recursion and a left node & right node
    # with all child nodes connected
    def __split_recursion(self, data_set, max_depth, max_estimator, split_info=None, ):
        # test if max_depth has proper value
        if max_depth is not None and max_depth <= 0:
            raise NotImplementedError('Tree Depth <=0')
            return None

        # get split info and the split datase
        split_info = self.__get_split_info(data_set)
        left_set, right_set, split_info = self.__split_data(data_set, split_info, max_estimator)
        self.tree_depth += 1
        print(f'tree_depth:{self.tree_depth} level is split...')

        # left branch can be split but right branch cannot
        if type(left_set) is not str and type(right_set) is str:
            # max_depth limit is not reached
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = (max_depth - 1) if (max_depth is not None) else None
                res = self.__split_recursion(left_set, max_depth, max_estimator, split_info, )
                if len(res) > 0:
                    child_left_node = res[0]
                    current_split_info = deepcopy(split_info)
                    # mother node data
                    mother_node_data = {
                        'type': 'non-leaf node',
                        '<': current_split_info['<'],
                        'residual': current_split_info['residual'],
                        'feature': current_split_info['feature'],
                        'sample': current_split_info['sample'],
                        'value': current_split_info['value'],
                        'lr_avg_target': current_split_info['lr_avg_target'],
                        'lr_residual': current_split_info['lr_residual'],
                    }
                    # right leaf child node data
                    child_right_node_data = {
                        'type': 'leaf node',
                        'residual': current_split_info['lr_residual'][1],
                        'sample': current_split_info['value'][1],
                        'class': current_split_info['lr_avg_target'][1],
                        'pos': 'right'
                    }
                    self.leaf_node_count += 1
                    # left child node data
                    child_left_node.data['pos'] = 'left'
                # links nodes together
                mother_node = TreeNode(mother_node_data)
                child_right_node = TreeNode(child_right_node_data)
                mother_node.set_left(child_left_node)
                mother_node.set_right(child_right_node)
                return (mother_node, split_info)
            # max_depth limit is reached
            else:
                # mother node data
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': split_info['<'],
                    'residual': split_info['residual'],
                    'feature': split_info['feature'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'lr_avg_target': split_info['lr_avg_target'],
                    'lr_residual': split_info['lr_residual'],
                }
                # left node data
                left_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['lr_residual'][0],
                    'sample': split_info['value'][0],
                    'class': split_info['lr_avg_target'][0],
                    'pos': 'left',
                }
                self.leaf_node_count += 1
                # right node data
                right_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['lr_residual'][1],
                    'sample': split_info['value'][1],
                    'class': split_info['lr_avg_target'][1],
                    'pos': 'right',
                }
                self.leaf_node_count += 1
                # build node and link all nodes together
                mother_node = TreeNode(mother_node_data)
                left_node = TreeNode(left_node_data)
                right_node = TreeNode(right_node_data)
                mother_node.set_left(left_node)
                mother_node.set_right(right_node)
                return (mother_node, split_info)

        # left branch cannot be split but right branch can
        elif type(left_set) is str and type(right_set) is not str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth != None else None
                res = self.__split_recursion(right_set, max_depth, max_estimator, split_info)
                if len(res) > 0:
                    child_right_node = res[0]
                    current_split_info = deepcopy(split_info)
                    mother_node_data = {
                        'type': 'non-leaf node',
                        '<': current_split_info['<'],
                        'residual': current_split_info['residual'],
                        'feature': current_split_info['feature'],
                        'sample': current_split_info['sample'],
                        'value': current_split_info['value'],
                        'lr_avg_target': current_split_info['lr_avg_target'],
                        'lr_residual': current_split_info['lr_residual'],
                    }
                    child_left_node_data = {
                        'type': 'leaf node',
                        'residual': current_split_info['lr_residual'][0],
                        'sample': current_split_info['value'][0],
                        'class': current_split_info['lr_avg_target'][0],
                        'pos': 'left',
                    }
                    self.leaf_node_count += 1
                    child_right_node.data['pos'] = 'right'
                    mother_node = TreeNode(mother_node_data)
                    child_left_node = TreeNode(child_left_node_data)
                    mother_node.set_left(child_left_node)
                    mother_node.set_right(child_right_node)
                    return (mother_node, split_info)
            else:
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': split_info['<'],
                    'residual': split_info['residual'],
                    'feature': split_info['feature'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'lr_avg_target': split_info['lr_avg_target'],
                    'lr_residual': split_info['lr_residual'],
                }
                left_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['lr_residual'][0],
                    'sample': split_info['value'][0],
                    'class': split_info['lr_avg_target'][0],
                    'pos': 'left',
                }
                self.leaf_node_count += 1
                right_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['lr_residual'][1],
                    'sample': split_info['value'][1],
                    'class': split_info['lr_avg_target'][1],
                    'pos': 'right',
                }
                self.leaf_node_count += 1
                mother_node = TreeNode(mother_node_data)
                left_node = TreeNode(left_node_data)
                right_node = TreeNode(right_node_data)
                mother_node.set_left(left_node)
                mother_node.set_right(right_node)

                return (mother_node, split_info)

        # both branches can be split
        elif type(left_set) is not str and type(right_set) is not str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth is not None else None
                # split left branch
                res_left = self.__split_recursion(left_set, max_depth, max_estimator, split_info, )
                if len(res_left) > 0:
                    left_child_node = res_left[0]
                    recursion_split_info_left = res_left[-1]
                # split right branch
                res_right = self.__split_recursion(right_set, max_depth, max_estimator, split_info)
                if len(res_right) > 0:
                    right_child_node = res_right[0]
                    recursion_split_info_right = res_right[-1]
                current_split_info = split_info
                # mother node data
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': current_split_info['<'],
                    'residual': current_split_info['residual'],
                    'feature': current_split_info['feature'],
                    'sample': current_split_info['sample'],
                    'value': current_split_info['value'],
                    'lr_avg_target': current_split_info['lr_avg_target'],
                    'lr_residual': current_split_info['lr_residual'],
                }

                mother_node = TreeNode(mother_node_data)
                mother_node.set_left(left_child_node)
                mother_node.set_right(right_child_node)

                return (mother_node, split_info)

            else:
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': split_info['<'],
                    'residual': split_info['residual'],
                    'feature': split_info['feature'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'lr_avg_target': split_info['lr_avg_target'],
                    'lr_residual': split_info['lr_residual'],
                }
                left_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['lr_residual'][0],
                    'sample': split_info['value'][0],
                    'class': split_info['lr_avg_target'][0],
                    'pos': 'left',
                }
                self.leaf_node_count += 1
                right_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['lr_residual'][1],
                    'sample': split_info['value'][1],
                    'class': split_info['lr_avg_target'][1],
                    'pos': 'right',
                }
                self.leaf_node_count += 1
                mother_node = TreeNode(mother_node_data)
                left_node = TreeNode(left_node_data)
                right_node = TreeNode(right_node_data)
                mother_node.set_left(left_node)
                mother_node.set_right(right_node)

                return (mother_node, split_info)

        # both branches cannot be split
        elif type(left_set) is str and type(right_set) is str:
            # split info is leaf node
            if split_info['type'] == 'leaf node':
                leaf_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['residual'],
                    'sample': split_info['sample'],
                    'class': split_info['lr_avg_target'],
                }
                self.leaf_node_count += 1
                leaf_node = TreeNode(leaf_node_data)
                return (leaf_node, split_info)
            # split info is not leaf node
            elif split_info['type'] == 'non-leaf node':
                mother_node_data = {
                    'type': 'non-leaf node',
                    '<': split_info['<'],
                    'residual': split_info['residual'],
                    'feature': split_info['feature'],
                    'sample': split_info['sample'],
                    'value': split_info['value'],
                    'lr_avg_target': split_info['lr_avg_target'],
                    'lr_residual': split_info['lr_residual'],
                }
                left_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['lr_residual'][0],
                    'sample': split_info['value'][0],
                    'class': split_info['lr_avg_target'][0],
                    'pos': 'left',
                }
                self.leaf_node_count += 1
                right_node_data = {
                    'type': 'leaf node',
                    'residual': split_info['lr_residual'][1],
                    'sample': split_info['value'][1],
                    'class': split_info['lr_avg_target'][1],
                    'pos': 'right',
                }
                self.leaf_node_count += 1
                mother_node = TreeNode(mother_node_data)
                left_node = TreeNode(left_node_data)
                right_node = TreeNode(right_node_data)
                mother_node.set_left(left_node)
                mother_node.set_right(right_node)
                return (mother_node, split_info)

    # calculate gini of all features of dataset to get the split feature info
    # data_set must have data & target
    # multiclass dataset target must modify as 0,1,2,3...
    def __get_split_info(self, data_set):
        if data_set != 'True' and data_set != 'False':
            feature_residual_lst = []
            for col_pos in range(len(data_set[0]) - 1):
                # print(f'column position:{col_pos}')
                feature_residual_lst.append(self.__calc_feature_residual(data_set, col_pos))
                if feature_residual_lst[-1] is None:
                    raise NotImplementedError('get_split_info func error: feature residual is None...')
                    break
        else:
            return data_set
        return sorted(feature_residual_lst, key=lambda val: val['residual'])[0]

    # split dataset based on the split_info for one time
    def __split_data(self, dataset, split_info, max_estimator):
        limit = 1 if max_estimator is None else max_estimator
        left_set = []
        right_set = []
        col_pos = split_info['feature']
        if col_pos is not None:
            col_min = [min(val) for val in list(zip(*dataset))][col_pos]
            col_max = [max(val) for val in list(zip(*dataset))][col_pos]
        # left branch cannot be split but right branch can be split
        if split_info['value'][0] <= limit and split_info['value'][1] > limit:
            left_set = 'pure'
            for index in range(len(dataset)):
                if split_info['<'] == col_max:
                    if dataset[index][split_info['feature']] >= split_info['<']:
                        right_set.append(dataset[index])
                else:
                    if dataset[index][split_info['feature']] > split_info['<']:
                        right_set.append(dataset[index])
        # right branch cannot be split but left branch can be split
        elif split_info['value'][1] <= limit and split_info['value'][0] > limit:
            right_set = 'pure'
            for index in range(len(dataset)):
                if split_info['<'] == col_min:
                    if dataset[index][split_info['feature']] <= split_info['<']:
                        left_set.append(dataset[index])
                else:
                    if dataset[index][split_info['feature']] < split_info['<']:
                        left_set.append(dataset[index])
        # both right and left branches cannot be split
        elif split_info['value'][0] <= limit and split_info['value'][1] <= limit:
            left_set = 'pure'
            right_set = 'pure'
        # both right and left branches can be split
        else:
            for index in range(len(dataset)):
                # split data is between max and min
                if split_info['<'] != col_min and split_info['<'] != col_max:
                    if dataset[index][split_info['feature']] <= split_info['<']:
                        left_set.append(dataset[index])
                    else:

                        right_set.append(dataset[index])
                # split data is min
                elif split_info['<'] == col_min:
                    if dataset[index][split_info['feature']] == split_info['<']:
                        left_set.append(dataset[index])
                    else:
                        right_set.append(dataset[index])
                # split data is max
                elif split_info['<'] == col_max:
                    if dataset[index][split_info['feature']] == split_info['<']:
                        right_set.append(dataset[index])
                    else:
                        left_set.append(dataset[index])
                # split data is both max & min
                elif split_info['<'] == col_max and split_info['<'] == col_min:
                    raise NotImplementedError('Only one value in split dataset..')
        return [left_set, right_set, split_info]

    # calculate gini for one feature for multiclass
    # return a split gini point
    # [total_gini, middle_value, feature_position, value_index, left_count, right_count]
    # multiclass target: 0, 1, 2, 3, 4,....
    # must modify the dataset target: 0,1,2,3,4....
    def __calc_feature_residual(self, data_set, pos):
        gini_lst = []
        data_sorted = deepcopy(sorted(data_set, key=lambda row: row[pos]))
        # data_sorted = deepcopy(data_set)
        col_max = [max(val) for val in list(zip(*data_set))][pos]
        col_min = [min(val) for val in list(zip(*data_set))][pos]
        # in a serial data column with proper max & min values
        if col_max > col_min:
            for index in range(len(data_sorted)):

                # if pos==3:
                #     print('found you!')

                left_count = []
                right_count = []
                if index + 1 == len(data_sorted):
                    break
                else:
                    # calc middle value between certain feature data
                    mid_val = (data_sorted[index][pos] + data_sorted[index + 1][pos]) / 2

                    # count class number around this middle value
                    for each in data_sorted:
                        if mid_val != col_min and mid_val != col_max and each[pos] != mid_val:
                            if each[pos] < mid_val:
                                left_count.append(each[-1])
                            elif each[pos] > mid_val:
                                right_count.append(each[-1])

                        elif mid_val == col_min:
                            if each[pos] <= mid_val:
                                left_count.append(each[-1])
                            else:
                                right_count.append(each[-1])

                        elif mid_val == col_max:
                            if each[pos] >= mid_val:
                                right_count.append(each[-1])
                            else:
                                left_count.append(each[-1])

                        elif each[pos] == mid_val:
                            left_count.append(each[-1])

                # the branches are not empty
                if len(left_count) != 0 and len(right_count) != 0:
                    # calc average target on both left & right branches
                    avg_left = sum(left_count) / len(left_count)
                    avg_right = sum(right_count) / len(right_count)

                    # calc residual of sorted data with average target
                    residual_left = 0
                    for val in left_count:
                        residual_left += (val - avg_left) ** 2
                    residual_right = 0
                    for val in right_count:
                        residual_right += (val - avg_right) ** 2
                    residual_sum = residual_left + residual_right
                else:
                    print(f'col_pos:{pos}')
                    print(f'tree level: {self.tree_depth}')
                    print('left or right count is none...')
                    print(data_set)
                    # print(*data_set, sep='\n')
                    return None

                total_count = len(left_count) + len(right_count)
                feature_position = pos
                result = {
                    '<': mid_val,
                    'type': 'non-leaf node',
                    "residual": residual_sum,
                    'feature': feature_position,
                    'sample': total_count,
                    'value': [len(left_count), len(right_count)],
                    'lr_avg_target': [avg_left, avg_right],
                    'lr_residual': [residual_left, residual_right],
                }
                gini_lst.append(result)

        # in a constant data column with same max & min values
        elif col_max == col_min:
            mid_val = (col_max + col_min) / 2
            class_count = [0]
            # count the target
            for data in data_sorted:
                class_count.append(data[-1])
            # calc target average & residual
            avg_class = sum(class_count) / len(class_count)
            residual_sum = 0
            for val in class_count:
                residual_sum += (val - avg_class) ** 2
            result = {
                '<': mid_val,
                'type': 'leaf node',
                'residual': residual_sum,
                'feature': None,
                'sample': sum(class_count),
                'value': [0, 0],
                'lr_avg_target': avg_class,
                'lr_residual': [0, 0],
            }
            gini_lst.append(result)

        return sorted(gini_lst, key=lambda val: val['residual'])[0]
