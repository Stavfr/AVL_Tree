# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info


"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type key: int or None
    @param key: key of your node
    @type value: any
    @param value: data of your node
    """

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = 0

    def create_leaf_with_virtual_nodes(self, key, value):
        leaf = AVLNode(key, value)
        leaf.set_height(0)
        leaf.set_size(1)
        leaf.set_left(AVLNode(None, None))
        leaf.set_right(AVLNode(None, None))

        return leaf

    def compute_balance_factor(self):
        if self == None or (not self.is_real_node()):
            return 0

        right_child_height = self.get_right().get_height()
        left_child_height = self.get_left().get_height()
        balance_factor = left_child_height - right_child_height

        self.height = max(right_child_height, left_child_height) + 1

        return balance_factor

    def is_empty_node(self, node):
        return node == None or (not node.is_real_node())

    def is_leaf(self):
        right_child = self.get_right()
        left_child = self.get_left()

        is_right_child_empty = self.is_empty_node(right_child)
        is_left_child_empty = self.is_empty_node(left_child)

        return is_right_child_empty and is_left_child_empty

    def has_one_child(self):
        right_child = self.get_right()
        left_child = self.get_left()

        is_right_child_empty = self.is_empty_node(right_child)
        is_left_child_empty = self.is_empty_node(left_child)

        return (is_right_child_empty and (not is_left_child_empty)) or \
            (is_left_child_empty and (not is_right_child_empty))

    def is_left_child_of_parent(self, node):
        parent = node.parent
        return parent.get_left() == node

    """returns the key

	@rtype: int or None
	@returns: the key of self, None if the node is virtual
	"""

    def get_key(self):
        return self.key

    """returns the value

	@rtype: any
	@returns: the value of self, None if the node is virtual
	"""

    def get_value(self):
        return self.value

    """returns the left child
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child (if self is virtual)
	"""

    def get_left(self):
        return self.left

    """returns the right child

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child (if self is virtual)
	"""

    def get_right(self):
        return self.right

    """returns the parent

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""

    def get_parent(self):
        return self.parent

    """returns the height

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""

    def get_height(self):
        return self.height

    """returns the size of the subtree

	@rtype: int
	@returns: the size of the subtree of self, 0 if the node is virtual
	"""

    def get_size(self):
        return self.height

    """sets key

	@type key: int or None
	@param key: key
	"""

    def set_key(self, key):
        # TODO: What is this?
        self.key = key

    """sets value

	@type value: any
	@param value: data
	"""

    def set_value(self, value):
        self.value = value

    """sets left child

	@type node: AVLNode
	@param node: a node
	"""

    def set_left(self, node):
        self.left = node
        node.set_parent(self)

    """sets right child

	@type node: AVLNode
	@param node: a node
	"""

    def set_right(self, node):
        self.right = node
        node.set_parent(self)

    """sets parent

	@type node: AVLNode
	@param node: a node
	"""

    def set_parent(self, node):
        new_parent: AVLNode = node
        self.parent = new_parent

        parent_left = new_parent.left
        parent_right = new_parent.right

        new_height = max(parent_left.height, parent_right.height) + 1
        new_parent.set_height(new_height)

        new_size = parent_left.height + parent_right.height + 1
        new_parent.set_size(new_size)

    """sets the height of the node

	@type h: int
	@param h: the height
	"""

    def set_height(self, h):
        self.height = h

    """sets the size of node

	@type s: int
	@param s: the size
	"""

    def set_size(self, s):
        self.size = s

    """returns whether self is not a virtual node

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""

    def is_real_node(self):
        return not self.key == None


"""
A class implementing an AVL tree.
"""


class AVLTree(object):

    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self):
        self.root = None
        # add your fields here

    def rotate_right(self, old_root: AVLNode):
        new_root: AVLNode = old_root.get_left()
        new_root_right = new_root.get_right()
        old_root.set_left(new_root_right)
        new_root.set_right = old_root

    def rotate_left(self, old_root: AVLNode):
        new_root: AVLNode = old_root.get_right()
        new_root_left = new_root.get_left()
        old_root.set_right(new_root_left)
        new_root.set_left = old_root

    def rotate_left_then_right(self, old_root: AVLNode):
        # Stands for A in the presentation
        left_old_new_root: AVLNode = old_root.get_left()

        self.rotate_left(left_old_new_root)
        self.rotate_right(old_root)

    def rotate_right_then_left(self, old_root: AVLNode):
        right_old_new_root: AVLNode = old_root.get_right()
        self.rotate_right(right_old_new_root)
        self.rotate_left(old_root)

    def find_successor_for_node_with_two_childs(self, node: AVLNode):
        successor_contestant: AVLNode = node.get_right()
        while not successor_contestant.get_left().is_empty_node(successor_contestant):
            successor_contestant = successor_contestant.get_left()
        return successor_contestant.get_parent()

    """searches for a value in the dictionary corresponding to the key

	@type key: int
	@param key: a key to be searched
	@rtype: any
	@returns: the value corresponding to key.
	"""

    def search(self, key):
        root = self.root

        while root != None and root.is_real_node():
            root_key = root.get_key()
            if root_key == key:
                return root
            if root_key < key:
                root = root.get_left()
            if root_key > key:
                root = root.get_right()

        return None

    """inserts val at position i in the dictionary

	@type key: int
	@pre: key currently does not appear in the dictionary
	@param key: key of item that is to be inserted to self
	@type val: any
	@param val: the value of the item
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

    def find_parent_for_insert(self, key):
        root: AVLNode = self.root

        while root.is_real_node():
            root_key = root.get_key()

            if root_key < key:
                root = root.get_left()
            if root_key > key:
                root = root.get_right()

        return root.get_parent()

    def find_parent_with_illegal_balance_factor(self, node: AVLNode):
        while node != None and abs(node.compute_balance_factor()) <= 1:
            node = node.parent
        return node

    def fix_tree_of_illegal_root(self, illegal_root):
        if illegal_root == None:
            return 0

        illegal_balance_factor = illegal_root.compute_balance_factor()

        if illegal_balance_factor == -2:
            right_child_balance_factor = illegal_root.get_right().compute_balance_factor()
            if right_child_balance_factor in [-1, 0]:
                self.rotate_left(illegal_root)
                return 1
            if right_child_balance_factor == 1:
                self.rotate_right_then_left(illegal_root)
                return 2
        else:  # illegal_balance_factor == 2
            left_child_balance_factor = illegal_root.get_left().compute_balance_factor()
            if left_child_balance_factor in [1, 0]:
                self.rotate_right(illegal_root)
                return 1
            if left_child_balance_factor == -1:
                self.rotate_left_then_right(illegal_root)
                return 2

    def physical_insert(self, leaf_for_insert: AVLNode):
        key = leaf_for_insert.get_key()
        parent_for_insert = self.find_parent_for_insert(key)
        parent_for_insert_key = parent_for_insert.get_key()

        if parent_for_insert_key > key:
            parent_for_insert.set_left(leaf_for_insert)
        if parent_for_insert_key < key:
            parent_for_insert.set_right(leaf_for_insert)

        return parent_for_insert

    def insert(self, key, val):
        leaf_for_insert = AVLNode.create_leaf_with_virtual_nodes(key, val)

        if self.root == None:
            self.root = leaf_for_insert

        parent_of_leaf = self.physical_insert(leaf_for_insert)
        node_with_illegal_balance_factor = self.find_parent_with_illegal_balance_factor(
            parent_of_leaf)

        balance_moves = self.fix_tree_of_illegal_root(
            node_with_illegal_balance_factor)

        return balance_moves

    """deletes node from the dictionary

	@type node: AVLNode
	@pre: node is a real pointer to a node in self
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

    def physical_delete(self, node: AVLNode):
        child = None
        left_child = node.get_left()
        if left_child.is_real_node():
            child = left_child
        else:
            child = node.get_right()

        node_parent = node.parent
        if node_parent == None:
            if child.is_real_node():
                self.root = child
            else:
                self.root = None
        else:
            if node.is_left_child_of_parent():
                node_parent.set_left(child)
            else:
                node_parent.set_right(child)
            node.parent = None

        return node_parent

    def replace_node_in_tree(self, new_node: AVLNode, old_node: AVLNode):
        if self.root == old_node:
            self.root = new_node

        new_node.set_left(old_node.get_left())
        new_node.set_right(old_node.get_right())
        new_node.parent = old_node.parent

    def delete(self, node: AVLNode):
        node_to_fix_from = None

        if node.is_leaf() or node.has_one_child():
            node_to_fix_from = self.physical_delete(node)
        else:
            successor = self.find_successor_for_node_with_two_childs(node)
            node_to_fix_from = self.physical_delete(successor)
            self.replace_node_in_tree(successor, node)

        sum = 0
        node_with_illegal_balance_factor = node_to_fix_from

        while node_with_illegal_balance_factor != None:
            node_with_illegal_balance_factor = self.find_parent_with_illegal_balance_factor(
                node_to_fix_from)

            sum += self.fix_tree_of_illegal_root(
                node_with_illegal_balance_factor)

        return sum

    """returns an array representing dictionary 

	@rtype: list
	@returns: a sorted list according to key of touples (key, value) representing the data structure
	"""

    def avl_to_array(self):
        return None

    """returns the number of items in dictionary 

	@rtype: int
	@returns: the number of items in dictionary 
	"""

    def size(self):
        return self.get_root().get_size()

    """splits the dictionary at a given node

	@type node: AVLNode
	@pre: node is in self
	@param node: The intended node in the dictionary according to whom we split
	@rtype: list
	@returns: a list [left, right], where left is an AVLTree representing the keys in the 
	dictionary smaller than node.key, right is an AVLTree representing the keys in the 
	dictionary larger than node.key.
	"""

    def split(self, node):
        return None

    """joins self with key and another AVLTree

	@type tree: AVLTree 
	@param tree: a dictionary to be joined with self
	@type key: int 
	@param key: The key separting self with tree
	@type val: any 
	@param val: The value attached to key
	@pre: all keys in self are smaller than key and all keys in tree are larger than key,
	or the other way around.
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""

    def join(self, tree, key, val):
        return None

    """compute the rank of node in the self

	@type node: AVLNode
	@pre: node is in self
	@param node: a node in the dictionary which we want to compute its rank
	@rtype: int
	@returns: the rank of node in self
	"""

    def rank(self, node):
        return None

    """finds the i'th smallest item (according to keys) in self

	@type i: int
	@pre: 1 <= i <= self.size()
	@param i: the rank to be selected in self
	@rtype: int
	@returns: the item of rank i in self
	"""

    def select(self, i):
        return None

    """returns the root of the tree representing the dictionary

	@rtype: AVLNode
	@returns: the root, None if the dictionary is empty
	"""

    def get_root(self):
        return None
