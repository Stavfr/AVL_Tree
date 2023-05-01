# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info

def printree(t, bykey=True):
    """Print a textual representation of t
    bykey=True: show keys instead of values"""
    # for row in trepr(t, bykey):
    #        print(row)
    return trepr(t, bykey)


def trepr(t, bykey=False):
    """Return a list of textual representations of the levels in t
    bykey=True: show keys instead of values"""
    if t == None:
        return ["#"]

    thistr = str(t.key) if bykey else str(t.val)

    return conc(trepr(t.left, bykey), thistr, trepr(t.right, bykey))


def conc(left, root, right):
    """Return a concatenation of textual represantations of
    a root node, its left node, and its right node
    root is a string, and left and right are lists of strings"""

    lwid = len(left[-1])
    rwid = len(right[-1])
    rootwid = len(root)

    result = [(lwid + 1) * " " + root + (rwid + 1) * " "]

    ls = leftspace(left[0])
    rs = rightspace(right[0])
    result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid *
                  " " + "\\" + rs * "_" + (rwid - rs) * " ")

    for i in range(max(len(left), len(right))):
        row = ""
        if i < len(left):
            row += left[i]
        else:
            row += lwid * " "

        row += (rootwid + 2) * " "

        if i < len(right):
            row += right[i]
        else:
            row += rwid * " "

        result.append(row)

    return result


def leftspace(row):
    """helper for conc"""
    # row is the first row of a left node
    # returns the index of where the second whitespace starts
    i = len(row) - 1
    while row[i] == " ":
        i -= 1
    return i + 1


def rightspace(row):
    """helper for conc"""
    # row is the first row of a right node
    # returns the index of where the first whitespace ends
    i = 0
    while row[i] == " ":
        i += 1
    return i


"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type key: int or None
    @param key: key of your node
    @type value: any
    @param value: data of your node
    """

    def __repr__(self):
        return "{}".format(self.key)

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = 0

    def create_leaf_with_virtual_nodes(key, value):
        leaf = AVLNode(key, value)
        leaf.set_height(0)
        leaf.set_size(1)
        leaf.left = AVLNode(None, None)
        leaf.left.parent = leaf
        leaf.right = AVLNode(None, None)
        leaf.right.parent = leaf

        return leaf

    def compute_balance_factor(self):
        if self == None or (not self.is_real_node()):
            return 0

        right_child_height = self.get_right().get_height()
        left_child_height = self.get_left().get_height()
        balance_factor = left_child_height - right_child_height

        self.height = max(right_child_height, left_child_height) + 1

        right_child_size = self.get_right().get_size()
        left_child_size = self.get_left().get_size()

        self.size = right_child_size + left_child_size + 1

        return balance_factor

    def is_empty_node(node):
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

    def is_left_child_of_parent(node):
        parent = node.get_parent()
        return parent.get_left() == node

    def update_parents_child(self, old_child, new_child):
        if AVLNode.is_left_child_of_parent(old_child):
            self.set_left(new_child)
        else:
            self.set_right(new_child)

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
        return self.size

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
        if node == None:
            self.parent = None
        else:
            new_parent: AVLNode = node
            self.parent = new_parent

            parent_left = new_parent.left
            parent_right = new_parent.right

            new_height = max(parent_left.get_height(),
                             parent_right.get_height()) + 1
            new_parent.set_height(new_height)

            new_size = parent_left.get_size() + parent_right.get_size() + 1
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

    def set_root(self, node):
        self.root = node
        node.set_parent(None)

    def rotate_right(self, old_root: AVLNode):
        print("Rotating right")
        new_root: AVLNode = old_root.get_left()
        new_root_right = new_root.get_right()

        old_root_parent = old_root.get_parent()
        if old_root_parent != None:
            old_root_parent.update_parents_child(old_root, new_root)
        else:
            self.set_root(new_root)

        old_root.set_left(new_root_right)
        new_root.set_right(old_root)

    def rotate_left(self, old_root: AVLNode):
        print("Rotating left")
        new_root: AVLNode = old_root.get_right()
        new_root_left = new_root.get_left()

        old_root_parent = old_root.get_parent()
        if old_root_parent != None:
            old_root_parent.update_parents_child(old_root, new_root)
        else:
            self.set_root(new_root)

        old_root.set_right(new_root_left)
        new_root.set_left(old_root)

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
        while not successor_contestant.is_empty_node(successor_contestant):
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
                root = root.get_right()
            if root_key > key:
                root = root.get_left()

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
                root = root.get_right()
            if root_key > key:
                root = root.get_left()

        return root.get_parent()

    def find_parent_with_illegal_balance_factor(self, node: AVLNode):
        while node != None and abs(node.compute_balance_factor()) <= 1:
            node = node.get_parent()
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

    def fix_tree(self, node_with_illegal_balance_factor):
        # When called 'node_with_illegal_balance_factor' is not necessarily illegal but it helps us for iterating using while
        sum = 0

        while node_with_illegal_balance_factor != None:
            node_with_illegal_balance_factor = self.find_parent_with_illegal_balance_factor(
                node_with_illegal_balance_factor)

            sum += self.fix_tree_of_illegal_root(
                node_with_illegal_balance_factor)

        return sum

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
            self.set_root(leaf_for_insert)
            return 0

        parent_of_leaf = self.physical_insert(leaf_for_insert)

        return self.fix_tree(parent_of_leaf)

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
                self.set_root(child)
            else:
                self.root = None
        else:
            if node.is_left_child_of_parent():
                node_parent.set_left(child)
            else:
                node_parent.set_right(child)
            node.parent = None

        return node_parent

    def replace_node_in_tree(self, old_node: AVLNode, new_node: AVLNode):
        if self.root == old_node:
            self.set_root(new_node)

        old_node_parent = old_node.get_parent()
        new_node.set_left(old_node.get_left())
        new_node.set_right(old_node.get_right())
        old_node_parent.update_parents_child(old_node, new_node)

    def delete(self, node: AVLNode):
        node_to_fix_from = None

        if node.is_leaf() or node.has_one_child():
            node_to_fix_from = self.physical_delete(node)
        else:
            successor = self.find_successor_for_node_with_two_childs(node)
            node_to_fix_from = self.physical_delete(successor)
            self.replace_node_in_tree(node, successor)

        return self.fix_tree(node_to_fix_from)

    """returns an array representing dictionary 

	@rtype: list
	@returns: a sorted list according to key of touples (key, value) representing the data structure
	"""

    def avl_to_array_rec(self, node: AVLNode, array):
        if AVLNode.is_empty_node(node):
            return
        self.avl_to_array_rec(node.get_left(), array)
        array.append((node.get_key(), node.get_value()))
        self.avl_to_array_rec(node.get_right(), array)

    def avl_to_array(self):
        array = []
        self.avl_to_array_rec(self.root, array)
        return array

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

    def join_tree_with_array_of_node_tuples(self, array_of_node_tuples):
        for mid_node, other_node in array_of_node_tuples:
            other_tree = self.create_tree(other_node)
            self.join(other_tree, mid_node.get_key(), mid_node.get_value())

    def split(self, node: AVLNode):
        greater_array_of_node_tuples = []
        lower_array_of_node_tuples = []
        greater_than_node_basis = node.get_right()
        greater_than_tree_basis = self.create_tree(greater_than_node_basis)
        lower_than_node_basis = node.get_left()
        lower_than_tree_basis = self.create_tree(lower_than_node_basis)
        parent = node.get_parent()

        while parent != None:
            if not node.is_left_child_of_parent():
                lower_array_of_node_tuples.append((parent, parent.get_left()))
            else:
                greater_array_of_node_tuples.append(
                    (parent, parent.get_right()))
            node = parent
            parent = parent.get_parent()

        lower_than_tree_basis.join_tree_with_array_of_node_tuples(
            lower_array_of_node_tuples)
        greater_than_tree_basis.join_tree_with_array_of_node_tuples(
            greater_array_of_node_tuples)

        return [lower_than_tree_basis, greater_than_tree_basis]

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

    def create_tree(self, root):
        tree = AVLTree()
        tree.set_root(root)
        return tree

    def join_trees_with_equal_heights(self, t1, t2, x):
        x.set_left(t1.root)
        x.set_right(t2.root)
        self.root = x

    def join_trees_left_tree_is_smaller(self, t1, t2, x):
        t2_node = t2.root

        while t2_node.get_height() > t1.root.get_height():
            t2_node = t2_node.get_left()

        b = t2_node
        b_parent = b.get_parent()

        x.set_left(t1.root)
        x.set_right(b)
        b_parent.set_left(x)

        t2.fix_tree(x)

        self.root = t2.root

    def join_trees_right_tree_is_smaller(self, t1, t2, x):
        t1_node = t1.root

        while t1_node.get_height() > t2.root.get_height():
            t1_node = t1_node.get_right()

        b = t1_node
        b_parent = b.get_parent()

        x.set_left(b)
        x.set_right(t2.root)
        b_parent.set_right(x)

        t1.fix_tree(x)

        self.root = t1.root

    def join(self, tree, key, val):
        if self.root.is_real_node():
            if self.root.get_key() < key:
                t1 = self
                t2 = tree
            else:
                t1 = tree
                t2 = self
        else:
            if tree.root.get_key() > key:
                t1 = self
                t2 = tree
            else:
                t1 = tree
                t2 = self

        t1_height = t1.root.get_height()
        t2_height = t2.root.get_height()
        height_delta = abs(t1_height - t2_height) + 1
        x = AVLNode.create_leaf_with_virtual_nodes(key, val)

        if t1_height == t2_height:
            self.join_trees_with_equal_heights(t1, t2, x)
            return height_delta

        if t1_height < t2_height:
            self.join_trees_left_tree_is_smaller(t1, t2, x)
            return height_delta

        self.join_trees_right_tree_is_smaller(t1, t2, x)
        return height_delta

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
        return self.root

    def __repr__(self):  # no need to understand the implementation of this one
        out = ""
        for row in printree(self.root):  # need printree.py file
            out = out + row + "\n"
        return out
