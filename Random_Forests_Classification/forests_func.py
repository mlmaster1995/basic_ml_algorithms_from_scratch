from decision_tree_node import TreeNode
from decision_tree import DecisionTree
from random_forest import RandomForest
import numpy as np
import pandas as pd
import random
import multiprocessing as mp


## print dictionary line by line
printdict = lambda dic: [print(key, ':', dic[key]) for key in dic]


### random forest build
# random patches: sampling both training instances and features -> sample columns & rows
# random subspaces: sampling features of all training instances -> sample columns
# patches_ratio: to sample ratio percentage of dataset randomly
# default tree_number is 100
'''
def build_forests_single(dataset, dataset_target, tree_number=None, max_depth=2, max_features=None, random_subspaces=True, patches_ratio= 0.7):
    random.seed = 4
    feature_length = len(dataset[0])
    dataset_height = len(dataset)
    patches_height = int(dataset_height*patches_ratio)
    dataset_frame = pd.DataFrame(dataset)
    forests_lst = []
    tree_features = []

    for i in range(100 if tree_number is None else tree_number):
        patches_dataset = []
        patches_dataset_target = []

        # get the sampling feature index & extract new data
        if max_features is not None and (max_features < 1 or max_features >= feature_length):
            raise NotImplementedError('improper sample features....')
        else:
            # extract features and combine all columns to new dataset
            feature_data_lst = []
            feature_index = []
            for j in range(2 if max_features is None else max_features):
                feature_pos = random.randrange(0, feature_length)
                feature_index.append(feature_pos)
                feature_data_lst.append(dataset_frame.iloc[:, feature_pos].tolist())
            new_dataset = np.column_stack(feature_data_lst).tolist()

            # if random_patches, randomly sample rows of new dataset with certain patches_ratio
            if not random_subspaces:
                for k in range(patches_height):
                    patches_pos = random.randrange(0, patches_height)
                    patches_dataset.append(new_dataset[patches_pos])
                    patches_dataset_target.append(dataset_target[patches_pos])

            # assign new_dataset & target to build trees
            new_dataset = new_dataset if random_subspaces else patches_dataset
            new_dataset_target = dataset_target if random_subspaces else patches_dataset_target

        # use new_dataset to build one new tree and save it in forests_lst
        tree_features.append(feature_index)
        decision_tree = build_tree(new_dataset, new_dataset_target, max_depth)
        forests_lst.append(decision_tree)

    # build random forest
    random_forests = RandomForest()
    random_forests.set_forest(forests_lst, tree_features)
    return random_forests
'''


def build_forests(dataset, dataset_target, tree_number=None, max_depth=2, max_features=None, random_subspaces=True, patches_ratio= 0.7, parallel=True,
                  multiclass=2):
    if not parallel:
        random.seed = 4
        feature_length = len(dataset[0])
        dataset_height = len(dataset)
        patches_height = int(dataset_height * patches_ratio)
        dataset_frame = pd.DataFrame(dataset)
        forests_lst = []
        tree_features = []

        for i in range(100 if tree_number is None else tree_number):
            patches_dataset = []
            patches_dataset_target = []

            # get the sampling feature index & extract new data
            if max_features is not None and (max_features < 1 or max_features >= feature_length):
                raise NotImplementedError('improper sample features....')
            else:
                # extract features and combine all columns to new dataset
                feature_data_lst = []
                feature_index = []
                for j in range(2 if max_features is None else max_features):
                    feature_pos = random.randrange(0, feature_length)
                    feature_index.append(feature_pos)
                    feature_data_lst.append(dataset_frame.iloc[:, feature_pos].tolist())
                new_dataset = np.column_stack(feature_data_lst).tolist()

                # if random_patches, randomly sample rows of new dataset with certain patches_ratio
                if not random_subspaces:
                    for k in range(patches_height):
                        patches_pos = random.randrange(0, patches_height)
                        patches_dataset.append(new_dataset[patches_pos])
                        patches_dataset_target.append(dataset_target[patches_pos])

                # assign new_dataset & target to build trees
                new_dataset = new_dataset if random_subspaces else patches_dataset
                new_dataset_target = dataset_target if random_subspaces else patches_dataset_target

            # use new_dataset to build one new tree and save it in forests_lst
            tree_features.append(feature_index)
            decision_tree = build_tree(new_dataset, new_dataset_target, max_depth, multiclass)
            forests_lst.append(decision_tree)
    else:
        forests_lst=[]
        tree_features=[]
        pool = mp.Pool(mp.cpu_count())
        forests_res = pool.starmap_async(_build_tree_advanced,
                                         [(dataset, dataset_target, max_depth, max_features, random_subspaces, patches_ratio, multiclass)
                                          for i in range(100 if tree_number is None else tree_number)]).get()
        pool.close()

        # extract trees & tree_features
        for item in forests_res:
            forests_lst.append(item[0])
            tree_features.append(item[1])

    # build random forest
    random_forests = RandomForest()
    random_forests.set_forest(forests_lst, tree_features)
    return random_forests


def _build_tree_advanced(dataset, dataset_target, max_depth, max_features, random_subspaces, patches_ratio, multiclass):
    random.seed = 4
    feature_length = len(dataset[0])
    dataset_height = len(dataset)
    patches_height = int(dataset_height*patches_ratio)
    dataset_frame = pd.DataFrame(dataset)

    patches_dataset = []
    patches_dataset_target = []

    # get the sampling feature index & extract new data
    if max_features is not None and (max_features < 1 or max_features >= feature_length):
        raise NotImplementedError('improper sample features....')
    else:
        # extract features and combine all columns to new dataset
        feature_data_lst = []
        feature_index = []
        for j in range(2 if max_features is None else max_features):
            feature_pos = random.randrange(0, feature_length)
            feature_index.append(feature_pos)
            feature_data_lst.append(dataset_frame.iloc[:, feature_pos].tolist())
        new_dataset = np.column_stack(feature_data_lst).tolist()

        # if random_patches, randomly sample rows of new dataset with certain patches_ratio
        if not random_subspaces:
            for k in range(patches_height):
                patches_pos = random.randrange(0, patches_height)
                patches_dataset.append(new_dataset[patches_pos])
                patches_dataset_target.append(dataset_target[patches_pos])

        # assign new_dataset & target to build trees
        new_dataset = new_dataset if random_subspaces else patches_dataset
        new_dataset_target = dataset_target if random_subspaces else patches_dataset_target
        decision_tree = build_tree(new_dataset, new_dataset_target, max_depth, multiclass)

    return [decision_tree, feature_index]

## build a tree
def build_tree(data_train, data_train_target, max_depth=None, multiclass=2):
    data_set = merge_data(data_train, data_train_target)
    res = split_recursion(data_set, multiclass, max_depth,)
    if res is not None:
        child_left_node = res[0]
        child_right_node = res[1]
        root_split_info = res[2]
        root_node_data={
           'type': 'root node',
           '<': root_split_info['<'],
           'gini': root_split_info['gini'],
           'feature': root_split_info['feature'],
           'sample': root_split_info['sample'],
           'value': root_split_info['value'],
           'lr_gini': root_split_info['lr_gini'],
           'lr_(Y|N)': root_split_info['lr_(Y|N)'],
           'pos': 'root',
         }
        root_node = TreeNode(root_node_data)
        root_node.set_left(child_left_node)
        root_node.set_right(child_right_node)

        decision_tree = DecisionTree()
        decision_tree.set_tree_root(root_node)
        return decision_tree
    else:
        decision_tree = DecisionTree()
        decision_tree.set_tree_root(None)
        return decision_tree


## split_the dataset repeatedly in a recursion and a left node & right node
## with all child nodes connected
'''
def split_recursion_original(data_set, multiclass, max_depth=None, split_info=None):
    if max_depth is not None and max_depth <= 0:
        raise NotImplementedError('Tree Depth <=0')
        return None

    split_info = get_split_info(data_set, multiclass)
    left_set, right_set, split_info = split_data(data_set, split_info)

    # gini is not zero but the dataset is pure
    if split_info['value'][0] == 0 or split_info['value'][1] == 0:
        non_gini_flag= True

    else:
        non_gini_flag = False

    # left branch can be split but right branch cannot
    if type(left_set) is not str and type(right_set) is str:
        if (max_depth is not None and max_depth-1 > 0) or max_depth is None:
            max_depth = max_depth-1 if max_depth is not None else None
            res = split_recursion(left_set, multiclass, max_depth, split_info)
            if len(res) == 3:
                child_node_left = res[0]
                child_node_right = res[1]
                now_split_info = res[2]

                new_left_node_data = {
                 'type': 'left node',
                 '<': now_split_info['<'],
                 'gini': now_split_info['gini'],
                 'feature': now_split_info['feature'],
                 'sample': now_split_info['sample'],
                 'value': now_split_info['value'],
                 'lr_gini': now_split_info['lr_gini'],
                 'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                 'pos': 'left',
                }
                new_right_node_data = {
                 'type': 'leaf node right',
                 'gini': split_info['lr_gini'][1],
                 'sample': split_info['value'][1],
                 'value': split_info['lr_(Y|N)'][1],
                 'pos': 'right',
                 'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
                }

            new_left_node = TreeNode(new_left_node_data)
            new_left_node.set_left(child_node_left)
            new_left_node.set_right(child_node_right)
            new_right_node = TreeNode(new_right_node_data)
            return (new_left_node, new_right_node, split_info)
        else:
            new_left_node_data = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
            }
            new_right_node_data = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
            }
            new_left_node = TreeNode(new_left_node_data)
            new_right_node = TreeNode(new_right_node_data)
            return(new_left_node, new_right_node, split_info)

    # left branch cannot be split but left branch can
    elif type(left_set) is str and type(right_set) is not str:
        if (max_depth is not None and max_depth-1>0) or max_depth is None:
            max_depth= max_depth-1 if max_depth != None else None
            res = split_recursion(right_set, multiclass, max_depth, split_info)
            if len(res) == 3:
                child_node_left= res[0]
                child_node_right= res[1]
                now_split_info = res[2]

                new_right_node_data={
                    'type':'right node',
                    '<': now_split_info['<'],
                    'gini': now_split_info['gini'],
                    'feature': now_split_info['feature'],
                    'sample': now_split_info['sample'],
                    'value': now_split_info['value'],
                    'lr_gini': now_split_info['lr_gini'],
                    'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                    'pos': 'right',
                }
                new_left_node_data={
                    'type': 'leaf node left',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_(Y|N)'][0],
                    'pos': 'left',
                    'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
                }

                new_right_node = TreeNode(new_right_node_data)
                new_right_node.set_left(child_node_left)
                new_right_node.set_right(child_node_right)
                new_left_node = TreeNode(new_left_node_data)
                return (new_left_node, new_right_node, split_info)
        else:
                new_left_node_data = {
                    'type': 'leaf node left',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_(Y|N)'][0],
                    'pos': 'left',
                    'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
                }
                new_right_node_data = {
                    'type': 'leaf node right',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_(Y|N)'][1],
                    'pos': 'right',
                    'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
                }
                new_left_node = TreeNode(new_left_node_data)
                new_right_node = TreeNode(new_right_node_data)
                return (new_left_node, new_right_node, split_info)

    # both branches can be split
    elif type(left_set) is not str and type(right_set) is not str:
        if (max_depth is not None and max_depth-1 > 0) or max_depth is None:
            max_depth = max_depth-1 if max_depth is not None else None
            # setup left side nodes
            res_left = split_recursion(left_set, multiclass, max_depth, split_info,)
            if len(res_left) == 3:
                child_node_left = res_left[0]
                child_node_right = res_left[1]
                now_split_info = res_left[2]

                new_left_node_data = {
                    'type': 'left node',
                    '<': now_split_info['<'],
                    'gini': now_split_info['gini'],
                    'feature': now_split_info['feature'],
                    'sample': now_split_info['sample'],
                    'value': now_split_info['value'],
                    'lr_gini': now_split_info['lr_gini'],
                    'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                    'pos': 'left',
                }

                new_left_node = TreeNode(new_left_node_data)
                new_left_node.set_left(child_node_left)
                new_left_node.set_right(child_node_right)

            res_right = split_recursion(right_set, multiclass, max_depth, split_info)
            if len(res_right) == 3:
                child_node_left= res_right[0]
                child_node_right= res_right[1]
                now_split_info = res_right[2]

                new_right_node_data={
                    'type':'right node',
                    '<': now_split_info['<'],
                    'gini': now_split_info['gini'],
                    'feature': now_split_info['feature'],
                    'sample':now_split_info['sample'],
                    'value':now_split_info['value'],
                    'lr_gini': now_split_info['lr_gini'],
                    'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                    'pos': 'right',
                }

                new_right_node = TreeNode(new_right_node_data)
                new_right_node.set_left(child_node_left)
                new_right_node.set_right(child_node_right)
                return (new_left_node, new_right_node, split_info)

        else:
            new_left_node_data = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
            }
            new_right_node_data = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
            }

            new_left_node = TreeNode(new_left_node_data)
            new_right_node = TreeNode(new_right_node_data)
            return (new_left_node, new_right_node, split_info)

    # both branches cannot be split
    elif type(left_set) is str and type(right_set) is str:
        data_left = {
        'type': 'leaf node left',
        'gini': split_info['lr_gini'][0],
        'sample': split_info['value'][0],
        'value': split_info['lr_(Y|N)'][0],
        'pos': 'left',
        'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
        }

        data_right = {
        'type': 'leaf node right',
        'gini': split_info['lr_gini'][1],
        'sample': split_info['value'][1],
        'value': split_info['lr_(Y|N)'][1],
        'pos': 'right',
        'class':1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
        }
        leaf_node_left = TreeNode(data_left)
        leaf_node_right = TreeNode(data_right)
        return (leaf_node_left, leaf_node_right, split_info)
def split_recursion_simple(data_set, multiclass, max_depth=None, split_info=None):
    if max_depth is not None and max_depth <= 0:
        raise NotImplementedError('Tree Depth <=0')
        return None

    split_info = get_split_info(data_set, multiclass)
    left_set, right_set, split_info = split_data(data_set, split_info)

    # it's a pure node, right_set is 'pure'
    if type(left_set) is not str and type(right_set) is str:

        new_left_node_data = {
            'type': 'leaf node left',
            'gini': split_info['lr_gini'][0],
            'sample': split_info['value'][0],
            'value': split_info['lr_(Y|N)'][0],
            'pos': 'left',
            'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
        }
        new_right_node_data = {
            'type': 'leaf node right',
            'gini': split_info['lr_gini'][1],
            'sample': split_info['value'][1],
            'value': split_info['lr_(Y|N)'][1],
            'pos': 'right',
            'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
        }
        new_left_node = TreeNode(new_left_node_data)
        new_right_node = TreeNode(new_right_node_data)
        return(new_left_node, new_right_node, split_info)

    # it's a pure node, left_set is 'pure'
    elif type(left_set) is str and type(right_set) is not str:

        new_left_node_data = {
            'type': 'leaf node left',
            'gini': split_info['lr_gini'][0],
            'sample': split_info['value'][0],
            'value': split_info['lr_(Y|N)'][0],
            'pos': 'left',
            'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
        }
        new_right_node_data = {
            'type': 'leaf node right',
            'gini': split_info['lr_gini'][1],
            'sample': split_info['value'][1],
            'value': split_info['lr_(Y|N)'][1],
            'pos': 'right',
            'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
        }
        new_left_node = TreeNode(new_left_node_data)
        new_right_node = TreeNode(new_right_node_data)
        return (new_left_node, new_right_node, split_info)

    # both branches can be split
    elif type(left_set) is not str and type(right_set) is not str:
        if (max_depth is not None and max_depth-1 > 0) or max_depth is None:
            max_depth = max_depth-1 if max_depth is not None else None
            # setup left side nodes
            res_left = split_recursion(left_set, multiclass, max_depth, split_info,)
            if len(res_left) == 3:
                child_node_left = res_left[0]
                child_node_right = res_left[1]
                now_split_info = res_left[2]

                new_left_node_data = {
                    'type': 'left node',
                    '<': now_split_info['<'],
                    'gini': now_split_info['gini'],
                    'feature': now_split_info['feature'],
                    'sample': now_split_info['sample'],
                    'value': now_split_info['value'],
                    'lr_gini': now_split_info['lr_gini'],
                    'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                    'pos': 'left',
                }

                new_left_node = TreeNode(new_left_node_data)
                new_left_node.set_left(child_node_left)
                new_left_node.set_right(child_node_right)

            res_right = split_recursion(right_set, multiclass, max_depth, split_info)
            if len(res_right) == 3:
                child_node_left= res_right[0]
                child_node_right= res_right[1]
                now_split_info = res_right[2]

                new_right_node_data={
                    'type':'right node',
                    '<': now_split_info['<'],
                    'gini': now_split_info['gini'],
                    'feature': now_split_info['feature'],
                    'sample':now_split_info['sample'],
                    'value':now_split_info['value'],
                    'lr_gini': now_split_info['lr_gini'],
                    'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                    'pos': 'right',
                }

                new_right_node = TreeNode(new_right_node_data)
                new_right_node.set_left(child_node_left)
                new_right_node.set_right(child_node_right)
                return (new_left_node, new_right_node, split_info)

        else:
            new_left_node_data = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
            }
            new_right_node_data = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
            }

            new_left_node = TreeNode(new_left_node_data)
            new_right_node = TreeNode(new_right_node_data)
            return (new_left_node, new_right_node, split_info)

    # left & right are 'pure' node
    elif type(left_set) is str and type(right_set) is str:
        data_left = {
        'type': 'leaf node left',
        'gini': split_info['lr_gini'][0],
        'sample': split_info['value'][0],
        'value': split_info['lr_(Y|N)'][0],
        'pos': 'left',
        'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
        }

        data_right = {
        'type': 'leaf node right',
        'gini': split_info['lr_gini'][1],
        'sample': split_info['value'][1],
        'value': split_info['lr_(Y|N)'][1],
        'pos': 'right',
        'class':1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
        }
        leaf_node_left = TreeNode(data_left)
        leaf_node_right = TreeNode(data_right)
        return (leaf_node_left, leaf_node_right, split_info)
'''
'''
def split_recursion_single(data_set, multiclass, max_depth=None, split_info=None):
    if max_depth is not None and max_depth <= 0:
        raise NotImplementedError('Tree Depth <=0')
        return None

    split_info = get_split_info(data_set, multiclass)
    left_set, right_set, split_info = split_data(data_set, split_info)

    # gini is not zero but the dataset is pure
    if split_info['value'][0] == 0 or split_info['value'][1] == 0:
        non_gini_flag= True
    else:
        non_gini_flag = False

    if not non_gini_flag:
        # left branch can be split but right branch cannot
        if type(left_set) is not str and type(right_set) is str:
            if (max_depth is not None and max_depth-1 > 0) or max_depth is None:
                max_depth = max_depth-1 if max_depth is not None else None
                res = split_recursion(left_set, multiclass, max_depth, split_info)
                if len(res) == 3:
                    child_node_left = res[0]
                    child_node_right = res[1]
                    now_split_info = res[2]

                    new_left_node_data = {
                     'type': 'left node',
                     '<': now_split_info['<'],
                     'gini': now_split_info['gini'],
                     'feature': now_split_info['feature'],
                     'sample': now_split_info['sample'],
                     'value': now_split_info['value'],
                     'lr_gini': now_split_info['lr_gini'],
                     'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                     'pos': 'left',
                    }
                    new_right_node_data = {
                     'type': 'leaf node right',
                     'gini': split_info['lr_gini'][1],
                     'sample': split_info['value'][1],
                     'value': split_info['lr_(Y|N)'][1],
                     'pos': 'right',
                     'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
                    }

                new_left_node = TreeNode(new_left_node_data)
                new_left_node.set_left(child_node_left)
                new_left_node.set_right(child_node_right)
                new_right_node = TreeNode(new_right_node_data)
                return (new_left_node, new_right_node, split_info)
            else:
                new_left_node_data = {
                    'type': 'leaf node left',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_(Y|N)'][0],
                    'pos': 'left',
                    'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
                }
                new_right_node_data = {
                    'type': 'leaf node right',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_(Y|N)'][1],
                    'pos': 'right',
                    'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
                }
                new_left_node = TreeNode(new_left_node_data)
                new_right_node = TreeNode(new_right_node_data)
                return(new_left_node, new_right_node, split_info)

        # left branch cannot be split but left branch can
        elif type(left_set) is str and type(right_set) is not str:
            if (max_depth is not None and max_depth-1>0) or max_depth is None:
                max_depth= max_depth-1 if max_depth != None else None
                res = split_recursion(right_set, multiclass, max_depth, split_info)
                if len(res) == 3:
                    child_node_left= res[0]
                    child_node_right= res[1]
                    now_split_info = res[2]

                    new_right_node_data={
                        'type':'right node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'right',
                    }
                    new_left_node_data={
                        'type': 'leaf node left',
                        'gini': split_info['lr_gini'][0],
                        'sample': split_info['value'][0],
                        'value': split_info['lr_(Y|N)'][0],
                        'pos': 'left',
                        'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
                    }

                    new_right_node = TreeNode(new_right_node_data)
                    new_right_node.set_left(child_node_left)
                    new_right_node.set_right(child_node_right)
                    new_left_node = TreeNode(new_left_node_data)
                    return (new_left_node, new_right_node, split_info)
            else:
                    new_left_node_data = {
                        'type': 'leaf node left',
                        'gini': split_info['lr_gini'][0],
                        'sample': split_info['value'][0],
                        'value': split_info['lr_(Y|N)'][0],
                        'pos': 'left',
                        'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
                    }
                    new_right_node_data = {
                        'type': 'leaf node right',
                        'gini': split_info['lr_gini'][1],
                        'sample': split_info['value'][1],
                        'value': split_info['lr_(Y|N)'][1],
                        'pos': 'right',
                        'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
                    }
                    new_left_node = TreeNode(new_left_node_data)
                    new_right_node = TreeNode(new_right_node_data)
                    return (new_left_node, new_right_node, split_info)

        # both branches can be split
        elif type(left_set) is not str and type(right_set) is not str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth is not None else None
                # setup left side nodes
                res_left = split_recursion(left_set, multiclass, max_depth, split_info, )
                if len(res_left) == 3:
                    child_node_left = res_left[0]
                    child_node_right = res_left[1]
                    now_split_info = res_left[2]

                    new_left_node_data = {
                        'type': 'left node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'left',
                    }

                    new_left_node = TreeNode(new_left_node_data)
                    new_left_node.set_left(child_node_left)
                    new_left_node.set_right(child_node_right)

                res_right = split_recursion(right_set, multiclass, max_depth, split_info)
                if len(res_right) == 3:
                    child_node_left = res_right[0]
                    child_node_right = res_right[1]
                    now_split_info = res_right[2]

                    new_right_node_data = {
                        'type': 'right node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'right',
                    }

                    new_right_node = TreeNode(new_right_node_data)
                    new_right_node.set_left(child_node_left)
                    new_right_node.set_right(child_node_right)
                    return (new_left_node, new_right_node, split_info)

            else:
                new_left_node_data = {
                    'type': 'leaf node left',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_(Y|N)'][0],
                    'pos': 'left',
                    'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
                }
                new_right_node_data = {
                    'type': 'leaf node right',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_(Y|N)'][1],
                    'pos': 'right',
                    'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
                }

                new_left_node = TreeNode(new_left_node_data)
                new_right_node = TreeNode(new_right_node_data)
                return (new_left_node, new_right_node, split_info)

        # both branches cannot be split
        elif type(left_set) is str and type(right_set) is str:
            data_left = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
            }

            data_right = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
            }
            leaf_node_left = TreeNode(data_left)
            leaf_node_right = TreeNode(data_right)
            return (leaf_node_left, leaf_node_right, split_info)

    else:
        # it's pure node, right_set is pure, but left_set cannot be split
        if type(left_set) is not str and type(right_set) is str:
            new_left_node_data = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': 1 if split_info['value'][0] > split_info['value'][1] else 0,
            }
            new_right_node_data = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'pos': 'right',
                'class': 1 if split_info['value'][1] > split_info['value'][0] else 0,
            }
            new_left_node = TreeNode(new_left_node_data)
            new_right_node = TreeNode(new_right_node_data)
            return (new_left_node, new_right_node, split_info)

        # it's pure node, left_set is pure, but right_set cannot be split
        elif type(left_set) is str and type(right_set) is not str:
            new_left_node_data = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'pos': 'left',
                'class': 1 if split_info['value'][0] > split_info['value'][1] else 0,
            }
            new_right_node_data = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': 1 if split_info['value'][1] > split_info['value'][0] else 0,
            }
            new_left_node = TreeNode(new_left_node_data)
            new_right_node = TreeNode(new_right_node_data)
            return (new_left_node, new_right_node, split_info)

        # both branches can be split
        elif type(left_set) is not str and type(right_set) is not str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth is not None else None
                # setup left side nodes
                res_left = split_recursion(left_set, multiclass, max_depth, split_info, )
                if len(res_left) == 3:
                    child_node_left = res_left[0]
                    child_node_right = res_left[1]
                    now_split_info = res_left[2]

                    new_left_node_data = {
                        'type': 'left node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'left',
                    }

                    new_left_node = TreeNode(new_left_node_data)
                    new_left_node.set_left(child_node_left)
                    new_left_node.set_right(child_node_right)

                res_right = split_recursion(right_set, multiclass, max_depth, split_info)
                if len(res_right) == 3:
                    child_node_left = res_right[0]
                    child_node_right = res_right[1]
                    now_split_info = res_right[2]

                    new_right_node_data = {
                        'type': 'right node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'right',
                    }

                    new_right_node = TreeNode(new_right_node_data)
                    new_right_node.set_left(child_node_left)
                    new_right_node.set_right(child_node_right)
                    return (new_left_node, new_right_node, split_info)

            else:
                new_left_node_data = {
                    'type': 'leaf node left',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_(Y|N)'][0],
                    'pos': 'left',
                    'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
                }
                new_right_node_data = {
                    'type': 'leaf node right',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_(Y|N)'][1],
                    'pos': 'right',
                    'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
                }

                new_left_node = TreeNode(new_left_node_data)
                new_right_node = TreeNode(new_right_node_data)
                return (new_left_node, new_right_node, split_info)

        # both branches cannot be split
        elif type(left_set) is str and type(right_set) is str:
            data_left = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': 1 if split_info['lr_(Y|N)'][0][0] > split_info['lr_(Y|N)'][0][1] else 0,
            }

            data_right = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': 1 if split_info['lr_(Y|N)'][1][0] > split_info['lr_(Y|N)'][1][1] else 0,
            }
            leaf_node_left = TreeNode(data_left)
            leaf_node_right = TreeNode(data_right)
            return (leaf_node_left, leaf_node_right, split_info)
'''

def split_recursion(data_set, multiclass, max_depth=None, split_info=None):
    if max_depth is not None and max_depth <= 0:
        raise NotImplementedError('Tree Depth <=0')
        return None

    split_info = get_split_info(data_set, multiclass)
    left_set, right_set, split_info = split_data(data_set, split_info)

    # gini is not zero but the dataset is pure
    if split_info['value'][0] == 0 or split_info['value'][1] == 0:
        non_gini_flag= True
    else:
        non_gini_flag = False

    # followed by non_gini_flag
    if not non_gini_flag:
        # left branch can be split but right branch cannot
        if type(left_set) is not str and type(right_set) is str:
            if (max_depth is not None and max_depth-1 > 0) or max_depth is None:
                max_depth = max_depth-1 if max_depth is not None else None
                res = split_recursion(left_set, multiclass, max_depth, split_info)
                if len(res) == 3:
                    child_node_left = res[0]
                    child_node_right = res[1]
                    now_split_info = res[2]

                    new_left_node_data = {
                     'type': 'left node',
                     '<': now_split_info['<'],
                     'gini': now_split_info['gini'],
                     'feature': now_split_info['feature'],
                     'sample': now_split_info['sample'],
                     'value': now_split_info['value'],
                     'lr_gini': now_split_info['lr_gini'],
                     'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                     'pos': 'left',
                    }
                    new_right_node_data = {
                     'type': 'leaf node right',
                     'gini': split_info['lr_gini'][1],
                     'sample': split_info['value'][1],
                     'value': split_info['lr_(Y|N)'][1],
                     'pos': 'right',
                     'class': sorted(range(len(split_info['lr_(Y|N)'][1])),
                                     key=lambda x: split_info['lr_(Y|N)'][1][x], reverse=False)[0],
                    }

                new_left_node = TreeNode(new_left_node_data)
                new_left_node.set_left(child_node_left)
                new_left_node.set_right(child_node_right)
                new_right_node = TreeNode(new_right_node_data)
                return (new_left_node, new_right_node, split_info)
            else:
                new_left_node_data = {
                    'type': 'leaf node left',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_(Y|N)'][0],
                    'pos': 'left',
                    'class': sorted(range(len(split_info['lr_(Y|N)'][0])),
                                     key=lambda x: split_info['lr_(Y|N)'][0][x], reverse=False)[0],
                }
                new_right_node_data = {
                    'type': 'leaf node right',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_(Y|N)'][1],
                    'pos': 'right',
                    'class': sorted(range(len(split_info['lr_(Y|N)'][1])),
                                     key=lambda x: split_info['lr_(Y|N)'][1][x], reverse=False)[0],
                }
                new_left_node = TreeNode(new_left_node_data)
                new_right_node = TreeNode(new_right_node_data)
                return(new_left_node, new_right_node, split_info)

        # left branch cannot be split but right branch can
        elif type(left_set) is str and type(right_set) is not str:
            if (max_depth is not None and max_depth-1>0) or max_depth is None:
                max_depth= max_depth-1 if max_depth != None else None
                res = split_recursion(right_set, multiclass, max_depth, split_info)
                if len(res) == 3:
                    child_node_left= res[0]
                    child_node_right= res[1]
                    now_split_info = res[2]

                    new_right_node_data={
                        'type':'right node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'right',
                    }
                    new_left_node_data={
                        'type': 'leaf node left',
                        'gini': split_info['lr_gini'][0],
                        'sample': split_info['value'][0],
                        'value': split_info['lr_(Y|N)'][0],
                        'pos': 'left',
                        'class': sorted(range(len(split_info['lr_(Y|N)'][0])),
                                     key=lambda x: split_info['lr_(Y|N)'][0][x], reverse=False)[0],
                    }

                    new_right_node = TreeNode(new_right_node_data)
                    new_right_node.set_left(child_node_left)
                    new_right_node.set_right(child_node_right)
                    new_left_node = TreeNode(new_left_node_data)
                    return (new_left_node, new_right_node, split_info)
            else:
                    new_left_node_data = {
                        'type': 'leaf node left',
                        'gini': split_info['lr_gini'][0],
                        'sample': split_info['value'][0],
                        'value': split_info['lr_(Y|N)'][0],
                        'pos': 'left',
                        'class': sorted(range(len(split_info['lr_(Y|N)'][0])),
                                     key=lambda x: split_info['lr_(Y|N)'][0][x], reverse=False)[0],
                    }
                    new_right_node_data = {
                        'type': 'leaf node right',
                        'gini': split_info['lr_gini'][1],
                        'sample': split_info['value'][1],
                        'value': split_info['lr_(Y|N)'][1],
                        'pos': 'right',
                        'class': sorted(range(len(split_info['lr_(Y|N)'][1])),
                                     key=lambda x: split_info['lr_(Y|N)'][1][x], reverse=False)[0],
                    }
                    new_left_node = TreeNode(new_left_node_data)
                    new_right_node = TreeNode(new_right_node_data)
                    return (new_left_node, new_right_node, split_info)

        # both branches can be split
        elif type(left_set) is not str and type(right_set) is not str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth is not None else None
                # setup left side nodes
                res_left = split_recursion(left_set, multiclass, max_depth, split_info, )
                if len(res_left) == 3:
                    child_node_left = res_left[0]
                    child_node_right = res_left[1]
                    now_split_info = res_left[2]

                    new_left_node_data = {
                        'type': 'left node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'left',
                    }

                    new_left_node = TreeNode(new_left_node_data)
                    new_left_node.set_left(child_node_left)
                    new_left_node.set_right(child_node_right)

                res_right = split_recursion(right_set, multiclass, max_depth, split_info)
                if len(res_right) == 3:
                    child_node_left = res_right[0]
                    child_node_right = res_right[1]
                    now_split_info = res_right[2]

                    new_right_node_data = {
                        'type': 'right node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'right',
                    }

                    new_right_node = TreeNode(new_right_node_data)
                    new_right_node.set_left(child_node_left)
                    new_right_node.set_right(child_node_right)
                    return (new_left_node, new_right_node, split_info)

            else:
                new_left_node_data = {
                    'type': 'leaf node left',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_(Y|N)'][0],
                    'pos': 'left',
                    'class': sorted(range(len(split_info['lr_(Y|N)'][0])),
                                     key=lambda x: split_info['lr_(Y|N)'][0][x], reverse=False)[0],
                }
                new_right_node_data = {
                    'type': 'leaf node right',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_(Y|N)'][1],
                    'pos': 'right',
                    'class': sorted(range(len(split_info['lr_(Y|N)'][1])),
                                     key=lambda x: split_info['lr_(Y|N)'][1][x], reverse=False)[0],
                }

                new_left_node = TreeNode(new_left_node_data)
                new_right_node = TreeNode(new_right_node_data)
                return (new_left_node, new_right_node, split_info)

        # both branches cannot be split
        elif type(left_set) is str and type(right_set) is str:
            data_left = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': sorted(range(len(split_info['lr_(Y|N)'][0])),
                                     key=lambda x: split_info['lr_(Y|N)'][0][x], reverse=False)[0],
            }

            data_right = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': sorted(range(len(split_info['lr_(Y|N)'][1])),
                                     key=lambda x: split_info['lr_(Y|N)'][1][x], reverse=False)[0],
            }
            leaf_node_left = TreeNode(data_left)
            leaf_node_right = TreeNode(data_right)
            return (leaf_node_left, leaf_node_right, split_info)

    else:
        # it's pure node, right_set is pure, and left_set cannot be split
        if type(left_set) is not str and type(right_set) is str:
            new_left_node_data = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': sorted(range(len(split_info['lr_(Y|N)'][0])),
                                     key=lambda x: split_info['lr_(Y|N)'][0][x], reverse=False)[0],
            }
            new_right_node_data = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'pos': 'right',
                'class': sorted(range(len(split_info['value'])),
                                     key=lambda x: split_info['value'][x], reverse=False)[0],
            }
            new_left_node = TreeNode(new_left_node_data)
            new_right_node = TreeNode(new_right_node_data)
            return (new_left_node, new_right_node, split_info)

        # it's pure node, left_set is pure, but right_set cannot be split
        elif type(left_set) is str and type(right_set) is not str:
            new_left_node_data = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'],
                'pos': 'left',
                'class': sorted(range(len(split_info['value'])),
                                     key=lambda x: split_info['value'][x], reverse=False)[0],
            }
            new_right_node_data = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': sorted(range(len(split_info['lr_(Y|N)'][1])),
                                     key=lambda x: split_info['lr_(Y|N)'][1][x], reverse=False)[0],
            }
            new_left_node = TreeNode(new_left_node_data)
            new_right_node = TreeNode(new_right_node_data)
            return (new_left_node, new_right_node, split_info)

        # both branches can be split
        elif type(left_set) is not str and type(right_set) is not str:
            if (max_depth is not None and max_depth - 1 > 0) or max_depth is None:
                max_depth = max_depth - 1 if max_depth is not None else None
                # setup left side nodes
                res_left = split_recursion(left_set, multiclass, max_depth, split_info, )
                if len(res_left) == 3:
                    child_node_left = res_left[0]
                    child_node_right = res_left[1]
                    now_split_info = res_left[2]

                    new_left_node_data = {
                        'type': 'left node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'left',
                    }

                    new_left_node = TreeNode(new_left_node_data)
                    new_left_node.set_left(child_node_left)
                    new_left_node.set_right(child_node_right)

                res_right = split_recursion(right_set, multiclass, max_depth, split_info)
                if len(res_right) == 3:
                    child_node_left = res_right[0]
                    child_node_right = res_right[1]
                    now_split_info = res_right[2]

                    new_right_node_data = {
                        'type': 'right node',
                        '<': now_split_info['<'],
                        'gini': now_split_info['gini'],
                        'feature': now_split_info['feature'],
                        'sample': now_split_info['sample'],
                        'value': now_split_info['value'],
                        'lr_gini': now_split_info['lr_gini'],
                        'lr_(Y|N)': now_split_info['lr_(Y|N)'],
                        'pos': 'right',
                    }

                    new_right_node = TreeNode(new_right_node_data)
                    new_right_node.set_left(child_node_left)
                    new_right_node.set_right(child_node_right)
                    return (new_left_node, new_right_node, split_info)

            else:
                new_left_node_data = {
                    'type': 'leaf node left',
                    'gini': split_info['lr_gini'][0],
                    'sample': split_info['value'][0],
                    'value': split_info['lr_(Y|N)'][0],
                    'pos': 'left',
                    'class': sorted(range(len(split_info['lr_(Y|N)'][0])),
                                     key=lambda x: split_info['lr_(Y|N)'][0][x], reverse=False)[0],
                }
                new_right_node_data = {
                    'type': 'leaf node right',
                    'gini': split_info['lr_gini'][1],
                    'sample': split_info['value'][1],
                    'value': split_info['lr_(Y|N)'][1],
                    'pos': 'right',
                    'class': sorted(range(len(split_info['lr_(Y|N)'][1])),
                                     key=lambda x: split_info['lr_(Y|N)'][1][x], reverse=False)[0],
                }

                new_left_node = TreeNode(new_left_node_data)
                new_right_node = TreeNode(new_right_node_data)
                return (new_left_node, new_right_node, split_info)

        # both branches cannot be split
        elif type(left_set) is str and type(right_set) is str:
            data_left = {
                'type': 'leaf node left',
                'gini': split_info['lr_gini'][0],
                'sample': split_info['value'][0],
                'value': split_info['lr_(Y|N)'][0],
                'pos': 'left',
                'class': sorted(range(len(split_info['lr_(Y|N)'][0])),
                                     key=lambda x: split_info['lr_(Y|N)'][0][x], reverse=False)[0],
            }

            data_right = {
                'type': 'leaf node right',
                'gini': split_info['lr_gini'][1],
                'sample': split_info['value'][1],
                'value': split_info['lr_(Y|N)'][1],
                'pos': 'right',
                'class': sorted(range(len(split_info['lr_(Y|N)'][1])),
                                     key=lambda x: split_info['lr_(Y|N)'][1][x], reverse=False)[0],
            }
            leaf_node_left = TreeNode(data_left)
            leaf_node_right = TreeNode(data_right)
            return (leaf_node_left, leaf_node_right, split_info)



## split dataset based on the split_info for one time
def split_data(dataset, split_info):
    left_set = []
    right_set = []

    if split_info['lr_gini'][0] == 0 and split_info['lr_gini'][1] != 0:
        left_set='pure'
        for index in range(len(dataset)):
            if dataset[index][split_info['feature']] > split_info['<']:
                right_set.append(dataset[index])

    elif split_info['lr_gini'][1] == 0 and split_info['lr_gini'][0] != 0:
        right_set='pure'
        for index in range(len(dataset)):
            if dataset[index][split_info['feature']] <= split_info['<']:
                left_set.append(dataset[index])

    elif split_info['lr_gini'][0] == 0 and split_info['lr_gini'][1] == 0:
        left_set='pure'
        right_set='pure'

    else:
        for index in range(len(dataset)):
            if dataset[index][split_info['feature']] <= split_info['<']:
                left_set.append(dataset[index])
            else:
                right_set.append(dataset[index])

    return [left_set, right_set, split_info]


## merget the data & target
def merge_data(data_set, data_set_target):
    data_merge =[]
    for res in zip(data_set, data_set_target):
        data_merge.append(res[0]+[res[1]])
    # print(*data_merge, sep='\n')
    return data_merge


## calculate gini of all features of dataset to get the split feature info
# data_set must have data & target
# multiclass dataset target must modify as 0,1,2,3...
def get_split_info(data_set, multiclass_number=2,):

    if data_set != 'True' and data_set != 'False':
        feature_gini_lst = []
        for index in range(len(data_set[0]) - 1):
            feature_gini_lst.append(calc_feature_gini(data_set, index, multiclass=multiclass_number))
            if feature_gini_lst[-1] is None:
                break

        # print('feature gini:', *feature_gini_lst, sep='\n')
        # print('\nsplit:\n', sorted(feature_gini_lst, key=lambda val: val['tgini'])[0])
    else:
        return data_set
    return sorted(feature_gini_lst, key=lambda val: val['gini'])[0]


## calculate gini for one feature for multiclass
# return a split gini point
# [total_gini, middle_value, feature_position, value_index, left_count, right_count]
# multiclass target: 0, 1, 2, 3, 4,....
# must modify the dataset target: 0,1,2,3,4....
def calc_feature_gini(data_set, pos, multiclass=2):
    gini_lst = []
    data_sorted = sorted(data_set, key=lambda row: row[pos])
    # data_sorted = data_set
    # print('sorted data:', *data_sorted, sep='\n')

    for index in range(len(data_sorted)):
        left_count = [0]*multiclass  # left branch: [0]->yes, [1]->no
        right_count = [0]*multiclass  # right branch: [0]->yes, [1]->no

        if index + 1 == len(data_sorted):
            break
        else:
            mid_val = (data_sorted[index][pos] + data_sorted[index + 1][pos]) / 2
            for each in data_sorted:
                if each[pos] <= mid_val:
                    for target in range(multiclass):
                        if each[-1] == target:
                            left_count[target] += 1
                        else:
                            continue
                elif each[pos] > mid_val:
                    for target in range(multiclass):
                        if each[-1] == target:
                            right_count[target] += 1
                        else:
                            continue

                # if each[pos] <= mid_val and each[-1] == 1:
                #     left_count[0] += 1
                # elif each[pos] <= mid_val and each[-1] == 0:
                #     left_count[1] += 1
                # elif each[pos] > mid_val and each[-1] == 1:
                #     right_count[0] += 1
                # elif each[pos] > mid_val and each[-1] == 0:
                #     right_count[1] += 1

        if sum(left_count) != 0 and sum(right_count) != 0:
            sum_left=0
            sum_right=0
            for target in range(multiclass):
                sum_left+= (left_count[target] / sum(left_count)) ** 2
                sum_right+= (right_count[target] / sum(right_count)) ** 2
            left_gini = 1-sum_left
            right_gini = 1 -sum_right

            # left_gini = 1 - (left_count[0] / sum(left_count)) ** 2 - (left_count[1] / sum(left_count)) ** 2
            # right_gini = 1 - (right_count[0] / sum(right_count)) ** 2 - (right_count[1] / sum(right_count)) ** 2

        elif sum(left_count)==0:
            sum_right = 0
            for target in range(multiclass):
                sum_right += (right_count[target] / sum(right_count)) ** 2
            right_gini = 1 - sum_right

            # right_gini = 1 - (right_count[0] / sum(right_count)) ** 2 - (right_count[1] / sum(right_count)) ** 2
            left_gini=0

        elif sum(right_count)==0:
            sum_left = 0
            for target in range(multiclass):
                sum_left += (left_count[target] / sum(left_count)) ** 2
            left_gini = 1 - sum_left

            # left_gini = 1 - (left_count[0] / sum(left_count)) ** 2 - (left_count[1] / sum(left_count)) ** 2
            right_gini=0

        else:
            print('Error: division zero,', 'feature position:', pos, 'index:',index)
            return None

        total_count = sum(left_count) + sum(right_count)
        total_gini = (sum(left_count) / total_count) * left_gini + (sum(right_count) / total_count) * right_gini
        feature_position = pos

        result ={
            '<': mid_val,
            "gini": total_gini,
            'feature': feature_position,
            'sample': sum(left_count)+sum(right_count),
            'value': [sum(left_count), sum(right_count)],
            'lr_gini': [left_gini, right_gini],
            'lr_(Y|N)': [left_count, right_count],
        }
        gini_lst.append(result)

    # print('gini list:', *gini_lst, sep='\n')
    return sorted(gini_lst, key=lambda val: val['gini'])[0]






