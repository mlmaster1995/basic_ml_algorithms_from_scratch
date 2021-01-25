# Developed By Chris Young in 2019
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.__left = None  # smaller than the parent node
        self.__right = None  # bigger than the parent node

    def set_right(self, node):
        if isinstance(node, TreeNode) or node is None:
            self.__right = node
        else:
            raise TypeError("The 'right' node must be of type Node or None.")

    def set_left(self, node):
        if isinstance(node, TreeNode) or node is None:
            self.__left = node
        else:
            raise TypeError("The 'left' node must be of type Node or None.")

    def get_right(self):
        return self.__right

    def get_left(self):
        return self.__left

    def print_details(self):
        print(f"{self.data}")
